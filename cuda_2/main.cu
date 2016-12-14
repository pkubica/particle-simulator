#include <sys/time.h>
#include <cstdio>
#include <cmath>

#include "nbody.h"


int main(int argc, char **argv)
{
    FILE *fp;
    struct timeval t1, t2;
    int N;
    float dt;
    int steps;
    int thr_blc;

    if (argc != 7)
    {
        printf("Usage: nbody <N> <dt> <steps> <thr/blc> <input> <output>\n");
        exit(1);
    }
    N = atoi(argv[1]);
    dt = atof(argv[2]);
    steps = atoi(argv[3]);
    thr_blc = atoi(argv[4]);

    printf("N: %d\n", N);
    printf("dt: %f\n", dt);
    printf("steps: %d\n", steps);
    printf("threads/block: %d\n", thr_blc);

    // allocation memory on CPU
    t_particles particles_cpu;

	if (cudaHostAlloc((void **) &particles_cpu.pos, N * sizeof(t_pos), cudaHostAllocDefault) != cudaSuccess)
	{
		fprintf(stderr, "Failed allocation on CPU!\n");
		exit(1);
	}
	if (cudaHostAlloc((void **) &particles_cpu.vel, N * sizeof(t_vel), cudaHostAllocDefault) != cudaSuccess)
	{
		fprintf(stderr, "Failed allocation on CPU!\n");
		cudaFreeHost(particles_cpu.pos);
		exit(1);
	}
    
	// read particles from file
    fp = fopen(argv[5], "r");
    if (fp == NULL)
    {
        printf("Can't open file %s!\n", argv[2]);
		cudaFreeHost(particles_cpu.pos);
		cudaFreeHost(particles_cpu.vel);
        exit(1);
    }
	particles_read(fp, particles_cpu, N);
    fclose(fp);

    t_particles particles_gpu[2];
    for (int i = 0; i < 2; i++)
    {
        // allocation memory on GPU
		if (cudaMalloc((void **) &particles_gpu[i].pos, N * sizeof(t_pos)) != cudaSuccess)
		{
			fprintf(stderr, "Failed allocation on GPU!\n");
			cudaFreeHost(particles_cpu.pos);
			cudaFreeHost(particles_cpu.vel);
			if(i == 1)
			{
				cudaFree(particles_gpu[0].pos);
				cudaFree(particles_gpu[0].vel);
			}
			exit(1);
		}
		if (cudaMalloc((void **) &particles_gpu[i].vel, N * sizeof(t_vel)) != cudaSuccess)
		{
			fprintf(stderr, "Failed allocation on GPU!\n");
			cudaFreeHost(particles_cpu.pos);
			cudaFreeHost(particles_cpu.vel);
			cudaFree(particles_gpu[0].pos);
			if(i == 1)
			{
				cudaFree(particles_gpu[0].vel);
				cudaFree(particles_gpu[1].pos);
			}
			exit(1);
		}

        // copy memory to GPU
		if (cudaMemcpy(particles_gpu[i].pos, particles_cpu.pos, N * sizeof(t_pos), cudaMemcpyHostToDevice) != cudaSuccess)
		{
			fprintf(stderr, "Failed cudamemcopyP to device!\n");
			cudaFree(particles_gpu[0].pos);
			cudaFree(particles_gpu[0].vel);
			cudaFree(particles_gpu[1].pos);
			cudaFree(particles_gpu[1].vel);
			cudaFreeHost(particles_cpu.pos);
			cudaFreeHost(particles_cpu.vel);
			exit(1);
		}
		if (cudaMemcpy(particles_gpu[i].vel, particles_cpu.vel, N * sizeof(t_vel), cudaMemcpyHostToDevice) != cudaSuccess)
		{
			fprintf(stderr, "Failed cudamemcopyV to device!\n");
			cudaFree(particles_gpu[0].pos);
			cudaFree(particles_gpu[0].vel);
			cudaFree(particles_gpu[1].pos);
			cudaFree(particles_gpu[1].vel);
			cudaFreeHost(particles_cpu.pos);
			cudaFreeHost(particles_cpu.vel);
			exit(1);
		}
    }

	int blc_grd = (N + thr_blc - 1) / thr_blc;

	// computing
    gettimeofday(&t1, 0);
    for (int s = 0; s < steps; s++)
    {
		if(s % 2)
	    	particles_simulate<<<blc_grd,thr_blc, thr_blc * 4 * sizeof(float)>>>(particles_gpu[1], particles_gpu[0], N, dt);
		else
			particles_simulate<<<blc_grd,thr_blc, thr_blc * 4 * sizeof(float)>>>(particles_gpu[0], particles_gpu[1], N, dt); 
	}

	cudaDeviceSynchronize();
	gettimeofday(&t2, 0);

    // cas
    double t = (1000000.0 * (t2.tv_sec - t1.tv_sec) + t2.tv_usec - t1.tv_usec) / 1000000.0;
    printf("Time: %f s\n", t);

    // copy particles back to main memory
	int i = (steps % 2) == 0 ? 0 : 1; 

	if (cudaMemcpy(particles_cpu.pos, particles_gpu[i].pos, N * sizeof(t_pos), cudaMemcpyDeviceToHost) != cudaSuccess)
	{
		fprintf(stderr, "Failed cudamemcopy from device!\n");
		cudaFree(particles_gpu[0].pos);
		cudaFree(particles_gpu[0].vel);
		cudaFree(particles_gpu[1].pos);
		cudaFree(particles_gpu[1].vel);
		cudaFreeHost(particles_cpu.pos);
		cudaFreeHost(particles_cpu.vel);
		exit(1);
	}
	if (cudaMemcpy(particles_cpu.vel, particles_gpu[i].vel, N * sizeof(t_vel), cudaMemcpyDeviceToHost) != cudaSuccess)
	{
		fprintf(stderr, "Failed cudamemcopy from device!\n");
		cudaFree(particles_gpu[0].pos);
		cudaFree(particles_gpu[0].vel);
		cudaFree(particles_gpu[1].pos);
		cudaFree(particles_gpu[1].vel);
		cudaFreeHost(particles_cpu.pos);
		cudaFreeHost(particles_cpu.vel);
		exit(1);
	}

    // save particles to file
    fp = fopen(argv[6], "w");
    if (fp == NULL)
    {
        printf("Can't open file %s!\n", argv[6]);
		cudaFree(particles_gpu[0].pos);
		cudaFree(particles_gpu[0].vel);
		cudaFree(particles_gpu[1].pos);
		cudaFree(particles_gpu[1].vel);
		cudaFreeHost(particles_cpu.pos);
		cudaFreeHost(particles_cpu.vel);
        exit(1);
    }
    particles_write(fp, particles_cpu, N);
    fclose(fp);

	cudaFree(particles_gpu[0].pos);
	cudaFree(particles_gpu[0].vel);
	cudaFree(particles_gpu[1].pos);
	cudaFree(particles_gpu[1].vel);
	cudaFreeHost(particles_cpu.pos);
	cudaFreeHost(particles_cpu.vel);

    return 0;
}
