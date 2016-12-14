#include <cstdio>
#include <cmath>

#include "nbody.h"

int main(int argc, char **argv)
{
    int i;
    FILE *fp;

    t_particles particles;

    if (argc != 3)
    {
        printf("Usage: nbody <input> <output>\n");
        exit(1);
    }

    // read particles from file
    fp = fopen(argv[1], "r");
    if (fp == NULL)
    {
        printf("Can't open file %s!\n", argv[1]);
        exit(1);
    }
    particles_read(fp, &particles);
    fclose(fp);

    // print parameters
    printf("N: %d\n", N);
    printf("dt: %f\n", DT);
    printf("steps: %d\n", STEPS);

    // do computing
    particles_simulate(&particles);


    // write particles to file
    fp = fopen(argv[2], "w");
    if (fp == NULL)
    {
        printf("Can't open file %s!\n", argv[2]);
        exit(1);
    }
    particles_write(fp, &particles);
    fclose(fp);
    return 0;
}
