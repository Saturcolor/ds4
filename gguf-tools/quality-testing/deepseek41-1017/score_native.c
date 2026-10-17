/* Exact-token 1017 deep scorer adapter for a source-pinned DS4 build.
 * This uses public session APIs, with no tokenizer or inference-engine patch.
 * Optional images use the maintained public encoder and multimodal session APIs.
 */
#define _GNU_SOURCE
#include "ds4.h"
#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifndef DS41_BUILD_SOURCE
#define DS41_BUILD_SOURCE "source-export"
#endif

typedef struct { uint32_t n, m; int *ids, *positions, *targets; } packet;
static void fail(const char *message) { fprintf(stderr,"score_native: %s\n",message); exit(1); }
static uint32_t u32(FILE *file) {
    unsigned char b[4];
    if(fread(b,1,4,file)!=4)fail("truncated packet");
    return (uint32_t)b[0]|(uint32_t)b[1]<<8|(uint32_t)b[2]<<16|(uint32_t)b[3]<<24;
}
static uint64_t header_limit;
static uint64_t u64(FILE *file) {
    uint64_t low=u32(file);return low|((uint64_t)u32(file)<<32);
}
static void skip_bytes(FILE *file,uint64_t count) {
    off_t at=ftello(file);
    if(at<0||(uint64_t)at>header_limit||count>header_limit-(uint64_t)at||
       fseeko(file,(off_t)count,SEEK_CUR))fail("invalid or oversized GGUF metadata");
}
static void text_value(FILE *file,char *out,size_t cap) {
    uint64_t n=u64(file);
    if(!n||n>=cap||fread(out,1,(size_t)n,file)!=(size_t)n)fail("invalid GGUF identity string");
    if(memchr(out,0,(size_t)n))fail("embedded NUL in GGUF identity");
    out[n]=0;
}
static void skip_value(FILE *file,uint32_t kind,unsigned depth) {
    static const uint8_t sizes[]={1,1,2,2,4,4,4,1,0,0,8,8,8};
    if(kind>12||depth>4)fail("invalid GGUF metadata type");
    if(kind==8){skip_bytes(file,u64(file));return;}
    if(kind==9){
        uint32_t element=u32(file);uint64_t count=u64(file);
        if(element>12||count>10000000)fail("invalid GGUF metadata array");
        if(element!=8&&element!=9){skip_bytes(file,count*sizes[element]);return;}
        for(uint64_t i=0;i<count;i++)skip_value(file,element,depth+1);
        return;
    }
    skip_bytes(file,sizes[kind]);
}
static uint64_t check_model(const char *path,bool vision) {
    FILE *file=fopen(path,"rb");if(!file)fail("cannot open GGUF");
    if(fseeko(file,0,SEEK_END))fail("cannot inspect GGUF length");
    off_t length=ftello(file);if(length<24)fail("truncated GGUF");rewind(file);
    header_limit=(uint64_t)length<67108864?(uint64_t)length:67108864;
    char magic[4];if(fread(magic,1,4,file)!=4||memcmp(magic,"GGUF",4)||u32(file)!=3)fail("GGUF v3 required");
    uint64_t tensors=u64(file),count=u64(file);
    if(!tensors||tensors>100000||count>100000)fail("invalid GGUF dimensions");
    bool arch=false,revision=false,source=false;
    for(uint64_t i=0;i<count;i++){
        char key[4096];text_value(file,key,sizeof(key));uint32_t kind=u32(file);
        bool *seen=NULL;const char *expected=NULL;
        if(!strcmp(key,"general.architecture")){seen=&arch;expected=vision?"deepseek4-vision":"deepseek41";}
        else if(!strcmp(key,"general.source.revision")){seen=&revision;expected="1bae0e1088993b9b82de08f59de5be5a34e42170";}
        else if(!strcmp(key,"general.source.url")){seen=&source;expected="https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017";}
        if(seen){
            char value[512];if(*seen||kind!=8)fail("duplicate or invalid GGUF source identity");
            text_value(file,value,sizeof(value));if(strcmp(value,expected))fail("exact DeepSeek V4.1 Flash1017 GGUF required");*seen=true;
        }else skip_value(file,kind,0);
        if((uint64_t)ftello(file)>header_limit)fail("oversized GGUF metadata");
    }
    fclose(file);if(!arch||!revision||!source)fail("GGUF lacks exact1017 source identity");
    return tensors;
}

static void put32(FILE *file,uint32_t value) {
    unsigned char b[4]={(unsigned char)value,(unsigned char)(value>>8),(unsigned char)(value>>16),(unsigned char)(value>>24)};
    if(fwrite(b,1,4,file)!=4)fail("output write failed");
}
static packet load(const char *path) {
    FILE *file=fopen(path,"rb");if(!file)fail("cannot open input packet");
    char magic[8];if(fread(magic,1,8,file)!=8||memcmp(magic,"DS411017",8))fail("wrong packet identity");
    uint32_t vocab=u32(file);packet p={0};p.n=u32(file);p.m=u32(file);uint32_t reserved=u32(file);
    if(vocab!=129280||!p.n||p.n>131072||!p.m||p.m>p.n||reserved)fail("invalid packet dimensions");
    p.ids=malloc((size_t)p.n*sizeof(int));p.positions=malloc((size_t)p.m*sizeof(int));p.targets=malloc((size_t)p.m*sizeof(int));
    if(!p.ids||!p.positions||!p.targets)fail("out of memory");
    for(uint32_t i=0;i<p.n;i++){uint32_t value=u32(file);if(value>=vocab)fail("invalid input token");p.ids[i]=(int)value;}
    for(uint32_t i=0;i<p.m;i++){uint32_t value=u32(file);if(value>=p.n||(i&&value<=(uint32_t)p.positions[i-1]))fail("invalid scored position");p.positions[i]=(int)value;}
    for(uint32_t i=0;i<p.m;i++){uint32_t value=u32(file);if(value>=vocab)fail("invalid target token");p.targets[i]=(int)value;}
    if(fgetc(file)!=EOF)fail("trailing packet bytes");
    fclose(file);
    return p;
}
static void release(packet *p){free(p->ids);free(p->positions);free(p->targets);}

static ds4_vision_span *images(ds4_engine *engine,const packet *p,const char *path,
                              const char *output,size_t *count) {
    FILE *file=fopen(path,"rb");if(!file)fail("cannot open image manifest");
    char magic[8];if(fread(magic,1,8,file)!=8||memcmp(magic,"DS41IMG1",8))fail("wrong image manifest identity");
    uint32_t n=u32(file);if(!n||n>16)fail("invalid image count");
    ds4_vision_span *spans=calloc(n,sizeof(*spans));if(!spans)fail("out of memory");
    uint32_t previous_end=0;char error[512];
    for(uint32_t i=0;i<n;i++) {
        uint32_t start=u32(file),tokens=u32(file),height=u32(file),width=u32(file),length=u32(file);
        if(!tokens||start<previous_end||start>=p->n||tokens>p->n-start||
           start+tokens>(uint32_t)p->positions[0]+1||!height||!width||length<2||length>4096)
            fail("invalid image span");
        char *image=malloc((size_t)length+1);if(!image)fail("out of memory");
        if(fread(image,1,length,file)!=length)fail("truncated image path");
        image[length]=0;
        if(image[0]!='/'||memchr(image,0,length)||memchr(image,'\n',length)||memchr(image,'\t',length))fail("unsafe image path");
        ds4_vision_embedding embedding={0};
        if(!ds4_engine_vision_encode_file(engine,image,&embedding,error,sizeof(error)))fail(error);
        free(image);
        if(embedding.grid_height!=height||embedding.grid_width!=width||
           (uint64_t)embedding.token_count!=(uint64_t)height*width)fail("encoded image geometry differs");
        size_t name_length=strlen(output)+48;char *name=malloc(name_length);if(!name)fail("out of memory");
        snprintf(name,name_length,"%s.vision-%u.f32",output,i);
        FILE *features=fopen(name,"wbx");if(!features)fail("image features output exists or cannot be created");
        put32(features,embedding.token_count);put32(features,5120);put32(features,height);put32(features,width);
        size_t values=(size_t)embedding.token_count*5120;
        for(size_t j=0;j<values;j++)if(!isfinite(embedding.data[j]))fail("nonfinite image embedding");
        if(fwrite(embedding.data,sizeof(float),values,features)!=values||fclose(features))fail("cannot save image embedding");
        free(name);
        ds4_tokens generated={0};
        if(!ds4_prompt_append_vision(engine,&generated,&spans[i],&embedding,error,sizeof(error)))fail(error);
        if(generated.len!=(int)tokens||spans[i].embedding.token_count!=tokens)fail("image token span length differs");
        for(uint32_t j=0;j<tokens;j++)if(generated.v[j]!=p->ids[start+j])fail("image placeholder tokens differ");
        ds4_tokens_free(&generated);spans[i].token_start=start;previous_end=start+tokens;
        fprintf(stderr,"image=%u/%u token_start=%u image_tokens=%u\n",i+1,n,start,tokens);
    }
    if(fgetc(file)!=EOF)fail("trailing image manifest bytes");
    fclose(file);*count=n;return spans;
}

int main(int argc,char **argv) {
    if(argc==2&&!strcmp(argv[1],"--version")) {
        puts("score_native " DS41_BUILD_SOURCE " protocol1");return 0;
    }
    if(argc==3&&!strcmp(argv[1],"--vision-info")) {
        uint64_t tensors=check_model(argv[2],true);
        printf("{\"architecture\":\"deepseek4-vision\",\"model_revision\":\"1bae0e1088993b9b82de08f59de5be5a34e42170\",\"tensors\":%llu}\n",(unsigned long long)tensors);return 0;
    }
    if(argc==3&&!strcmp(argv[1],"--model-info")) {
        uint64_t tensors=check_model(argv[2],false);
        printf("{\"architecture\":\"deepseek41\",\"model_revision\":\"1bae0e1088993b9b82de08f59de5be5a34e42170\",\"tensors\":%llu}\n",(unsigned long long)tensors);return 0;
    }
    if(argc==2&&!strcmp(argv[1],"--help")) {
        puts("score_native " DS41_BUILD_SOURCE " protocol1\nUsage: score_native MODEL.gguf JOBS.tsv CONTEXT [VISION.gguf]\n       score_native --check-packet CASE.bin\nVision jobs use packet<TAB>output<TAB>image-manifest; existing public multimodal APIs.");return 0;
    }
    if(argc==3&&!strcmp(argv[1],"--check-packet")) {
        packet p=load(argv[2]);printf("{\"input_tokens\":%u,\"scored_positions\":%u,\"packet_valid\":true}\n",p.n,p.m);release(&p);return 0;
    }
    if(argc!=4&&argc!=5)fail("use --help");
    check_model(argv[1],false);if(argc==5)check_model(argv[4],true);
    char *end=NULL;long context=strtol(argv[3],&end,10);
    if(!end||*end||context<1024||context>131328)fail("invalid context limit");
    uint32_t endian=1;if(*(unsigned char *)&endian!=1||sizeof(float)!=4)fail("requires little-endian FP32 host");
    ds4_engine_options options={.model_path=argv[1],.vision_path=argc==5?argv[4]:NULL,.backend=DS4_BACKEND_CUDA,.n_threads=4,
                               .context_size=(int)context,.placement_ctx_hint=(int)context,
                               .placement_session_count_hint=1,.warm_weights=false,.quality=true};
    ds4_engine *engine=NULL;
    if(ds4_engine_open(&engine,&options))fail("cannot load 1017 model");
    if(ds4_engine_vocab_size(engine)!=129280||ds4_engine_layer_count(engine)!=40||
       ds4_engine_embd_dim(engine)!=5120||(ds4_engine_routed_quant_bits(engine)!=2 && ds4_engine_routed_quant_bits(engine)!=4))fail("wrong1017 model shape or unsupported routed quant");
    if(argc==5&&!ds4_engine_has_vision(engine))fail("vision encoder not loaded");
    fprintf(stderr,"model=%s prefill_chunk=%u power=%d source=%s\n",ds4_engine_model_name(engine),
            ds4_engine_prefill_chunk(engine),ds4_engine_power(engine),DS41_BUILD_SOURCE);
    ds4_session *session=NULL;if(ds4_session_create(&session,engine,(int)context))fail("cannot create scorer session");
    float *logits=malloc(129280*sizeof(float));if(!logits)fail("out of memory");
    FILE *jobs=fopen(argv[2],"rb");if(!jobs)fail("cannot read job manifest");
    char *line=NULL;size_t capacity=0;ssize_t length;unsigned completed=0;
    while((length=getline(&line,&capacity,jobs))>=0) {
        while(length&&(line[length-1]=='\n'||line[length-1]=='\r'))line[--length]=0;
        if(!length)continue;
        char *output=strchr(line,'\t');if(!output)fail("job needs packet<TAB>output");*output++=0;
        char *image_manifest=strchr(output,'\t');
        if(image_manifest)*image_manifest++=0;
        if((image_manifest!=NULL)!=(argc==5)||(image_manifest&&strchr(image_manifest,'\t')))fail("wrong image job columns");
        packet p=load(line);if((long)p.n>=context)fail("case exceeds scorer context");
        size_t image_count=0;
        ds4_vision_span *spans=image_manifest?images(engine,&p,image_manifest,output,&image_count):NULL;
        FILE *out=fopen(output,"wbx");if(!out)fail("output exists or cannot be created");
        if(fwrite("DS41OUT1",1,8,out)!=8)fail("output write failed");
        put32(out,129280);
        put32(out,p.m);
        ds4_session_invalidate(session);int previous=0;char error[512];
        for(uint32_t i=0;i<p.m;i++) {
            int wanted=p.positions[i]+1;
            if(previous&&wanted==previous+1) {
                if(ds4_session_eval(session,p.ids[wanted-1],error,sizeof(error)))fail(error);
            } else {
                ds4_tokens prefix={.v=p.ids,.len=wanted,.cap=(int)p.n};
                int status=image_count?ds4_session_sync_multimodal(session,&prefix,spans,image_count,error,sizeof(error)):
                                       ds4_session_sync(session,&prefix,error,sizeof(error));
                if(status)fail(error);
            }
            previous=wanted;
            if(ds4_session_copy_logits(session,logits,129280)!=129280)fail("incomplete logits");
            for(int j=0;j<129280;j++)if(!isfinite(logits[j]))fail("nonfinite candidate logits");
            put32(out,(uint32_t)p.positions[i]);put32(out,(uint32_t)p.targets[i]);
            if(fwrite(logits,sizeof(float),129280,out)!=129280)fail("output write failed");
            if(i%16==0||i+1==p.m)fprintf(stderr,"case=%u rows=%u/%u position=%d\n",completed+1,i+1,p.m,p.positions[i]);
        }
        if(fclose(out))fail("output close failed");
        ds4_session_invalidate(session);
        for(size_t i=0;i<image_count;i++)ds4_vision_embedding_free(&spans[i].embedding);
        free(spans);
        release(&p);
        completed++;
    }
    free(line);fclose(jobs);free(logits);ds4_session_free(session);ds4_engine_close(engine);
    fprintf(stderr,"complete_cases=%u\n",completed);return 0;
}
