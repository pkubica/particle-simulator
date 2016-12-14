#ifndef __NBODY_H__
#define __NBODY_H__

#include <cstdlib>
#include <cstdio>

#define G 6.67384e-11f

#define PADDING 32 - (N % 32)
#define ALIGN 32

typedef struct
{
	__declspec(align(ALIGN)) float pos_x[N + PADDING];
	__declspec(align(ALIGN)) float pos_y[N + PADDING];
	__declspec(align(ALIGN)) float pos_z[N + PADDING];
	__declspec(align(ALIGN)) float vel_x[N + PADDING];
	__declspec(align(ALIGN)) float vel_y[N + PADDING];
	__declspec(align(ALIGN)) float vel_z[N + PADDING];
	__declspec(align(ALIGN)) float weight[N + PADDING];
} t_particle;

typedef t_particle t_particles;

void particles_simulate(t_particles *p);

void particles_read(FILE *fp, t_particles *p);

void particles_write(FILE *fp, t_particles *p);

#endif
