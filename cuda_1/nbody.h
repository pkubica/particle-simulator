#ifndef __NBODY_H__
#define __NBODY_H__

#include <cstdlib>
#include <cstdio>

#define G 6.67384e-11f

typedef struct __align__(16)
{
	float x;
	float y;
	float z;
	float weight;
}t_pos;

typedef struct __align__(16)
{
	float x;
	float y;
	float z;
}t_vel;

typedef struct __align__(16)
{

	t_pos *pos;
	t_vel *vel;

}t_particles;

__global__ void particles_simulate(t_particles p_in, t_particles p_out, int N, float dt);

void particles_read(FILE *fp, t_particles &p, int N);

void particles_write(FILE *fp, t_particles &p, int N);

#endif
