/* Inspect the maintained V4.1 image preprocessing without loading a model. */
#include "ds4_image.h"
#ifndef DS41_BUILD_SOURCE
#define DS41_BUILD_SOURCE "source-export"
#endif
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void fail(const char *message) {
    fprintf(stderr, "image_probe: %s\n", message);
    exit(1);
}

int main(int argc, char **argv) {
    if (argc == 2 && !strcmp(argv[1], "--help")) {
        puts("image_probe " DS41_BUILD_SOURCE "\nUsage: image_probe IMAGE OUT_PREFIX");
        return 0;
    }
    if (argc != 3) fail("use --help");
    uint32_t endian = 1;
    if (*(uint8_t *)&endian != 1 || sizeof(float) != 4) fail("requires little-endian FP32");
    char error[512] = {0};
    ds4_image image = {0};
    ds4_deepseek4_image_patches patches = {0};
    ds4_deepseek4_image_layout layout = {0};
    if (!ds4_image_decode_file(&image, argv[1], error, sizeof(error))) fail(error);
    if (!ds4_image_preprocess_deepseek41(&patches, &image, error, sizeof(error))) fail(error);
    if (!ds4_deepseek41_image_layout_build(&layout, patches.llm_grid_height,
                                          patches.llm_grid_width, error, sizeof(error))) fail(error);
    size_t length = strlen(argv[2]) + 16;
    char *path = malloc(length);
    if (!path) fail("out of memory");
    snprintf(path, length, "%s.f32", argv[2]);
    FILE *file = fopen(path, "wbx");
    uint64_t values = (uint64_t)patches.patch_count * 3 * 14 * 14;
    if (!file || fwrite(patches.patches, sizeof(float), values, file) != values || fclose(file))
        fail("cannot save patch values");
    snprintf(path, length, "%s.types", argv[2]);
    file = fopen(path, "wbx");
    if (!file || fwrite(layout.types, 1, layout.token_count, file) != layout.token_count || fclose(file))
        fail("cannot save image layout");
    printf("{\"build_source\":\"%s\","
           "\"image_width\":%u,\"image_height\":%u,\"content_width\":%u,\"content_height\":%u,"
           "\"padded_width\":%u,\"padded_height\":%u,\"grid_width\":%u,\"grid_height\":%u,"
           "\"llm_grid_width\":%u,\"llm_grid_height\":%u,\"patch_count\":%u,"
           "\"patch_values\":%llu,\"image_token_count\":%u}\n",
           DS41_BUILD_SOURCE, image.width, image.height, patches.content_width, patches.content_height,
           patches.padded_width, patches.padded_height, patches.grid_width, patches.grid_height,
           patches.llm_grid_width, patches.llm_grid_height, patches.patch_count,
           (unsigned long long)values, layout.token_count);
    free(path);
    ds4_deepseek4_image_layout_free(&layout);
    ds4_deepseek4_image_patches_free(&patches);
    ds4_image_free(&image);
    return 0;
}
