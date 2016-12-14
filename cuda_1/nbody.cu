#include <cmath>
#include <cfloat>
#include "nbody.h"

#define PREVENCY_DIVISION_ZERO 1e-9f

__global__ void particles_simulate(t_particles p_in, t_particles p_out, int N, float dt)
{
	int i = threadIdx.x + blockDim.x * blockIdx.x;

	float dx, dy, dz, fract, tmp;
	float ax = 0, ay = 0, az = 0;
	const float current_x = p_in.pos[i].x;
	const float current_y = p_in.pos[i].y;
	const float current_z = p_in.pos[i].z;

	#pragma unroll 16
	for (int j = 0; j < N; ++j)
	{
		dx = p_in.pos[j].x - current_x;
		dy = p_in.pos[j].y - current_y;
		dz = p_in.pos[j].z - current_z;

		fract = dx * dx + dy * dy + dz * dz;
		
		fract += PREVENCY_DIVISION_ZERO;

		fract = rsqrtf(fract);
 
		tmp = dt * p_in.pos[j].weight * G * fract* fract* fract;

		ax += dx * tmp;
		ay += dy * tmp;
		az += dz * tmp;
	}

	float velx = p_in.vel[i].x;
	float vely = p_in.vel[i].y;
	float velz = p_in.vel[i].z;

	velx += ax;
	vely += ay;
	velz += az;

	p_out.vel[i].x = velx;
	p_out.vel[i].y = vely;
	p_out.vel[i].z = velz;

	p_out.pos[i].x = current_x + velx * dt;
	p_out.pos[i].y = current_y + vely * dt;
	p_out.pos[i].z = current_z + velz * dt;
}

void particles_read(FILE *fp, t_particles &p, int N)
{
    int i;
    for (i = 0; i < N; i++)
    {
        fscanf(fp, "%f %f %f %f %f %f %f \n", &p.pos[i].x, &p.pos[i].y, &p.pos[i].z, &p.vel[i].x, &p.vel[i].y, &p.vel[i].z, &p.pos[i].weight);
    }
}

void particles_write(FILE *fp, t_particles &p, int N)
{
    int i;
    for (i = 0; i < N; i++)
    {
        fprintf(fp, "%10.10f %10.10f %10.10f %10.10f %10.10f %10.10f %10.10f \n", p.pos[i].x, p.pos[i].y, p.pos[i].z, p.vel[i].x, p.vel[i].y, p.vel[i].z, p.pos[i].weight);
    }
}
