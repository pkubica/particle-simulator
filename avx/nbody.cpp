/*
 * Architektura procesoru (ACH 2015)
 * Projekt c. 1 (matvec)
 * Login: xkubic22
 */

#include <cfloat>
#include <cmath>
#include "nbody.h"
#include <iostream>



void particles_simulate(t_particles *p)
{
	__declspec(align(ALIGN)) float dx[N + PADDING];
	__declspec(align(ALIGN)) float dy[N + PADDING];
	__declspec(align(ALIGN)) float dz[N + PADDING];
	__declspec(align(ALIGN)) float sqrt_r[N + PADDING];
	__declspec(align(ALIGN)) float r[N + PADDING];

	__assume_aligned(&p->pos_x, 32);
	__assume_aligned(&p->pos_y, 32);
	__assume_aligned(&p->pos_z, 32);
	__assume_aligned(&p->vel_x, 32);
	__assume_aligned(&p->vel_y, 32);
	__assume_aligned(&p->vel_z, 32);
	__assume_aligned(&p->weight, 32);

	for (unsigned int step = 0; step < STEPS; ++step)
	{
		for (unsigned int j = 0; j < N + PADDING; ++j)
		{
			unsigned int offset = j / ALIGN;

#pragma vector aligned
#pragma ivdep
			for (unsigned int i = offset * ALIGN; i < N + PADDING; ++i) //o
			{
				dx[i] = p->pos_x[i] - p->pos_x[j]; //o
				dy[i] = p->pos_y[i] - p->pos_y[j]; //o
				dz[i] = p->pos_z[i] - p->pos_z[j]; //o

			}
#pragma vector aligned
#pragma ivdep
			for (unsigned int i = offset * ALIGN; i < N + PADDING; ++i) //o
			{
				r[i] = pow(dx[i] * dx[i] + dy[i] * dy[i] + dz[i] * dz[i], 3);  //o
			}

#pragma vector aligned
#pragma ivdep
			for (unsigned int i = offset * ALIGN; i < N + PADDING; ++i) //o
			{
				float sqr = sqrt(r[i]); //o
				sqrt_r[i] = sqr > 0 ? sqr : 1;
			}

#pragma vector aligned
#pragma ivdep
			for (unsigned int i = offset * ALIGN; i < N + PADDING; ++i) //o
			{
				p->vel_x[j] += p->weight[i] * dx[i] * G * DT / sqrt_r[i]; //o
				p->vel_y[j] += p->weight[i] * dy[i] * G * DT / sqrt_r[i]; //o
				p->vel_z[j] += p->weight[i] * dz[i] * G * DT / sqrt_r[i]; //o

				if (i > j)
				{
					p->vel_x[i] -= p->weight[j] * dx[i] * G * DT / sqrt_r[i]; //o
					p->vel_y[i] -= p->weight[j] * dy[i] * G * DT / sqrt_r[i]; //o
					p->vel_z[i] -= p->weight[j] * dz[i] * G * DT / sqrt_r[i]; //o
				}
				else
				{
					p->vel_x[i] -= 0;
					p->vel_y[i] -= 0;
					p->vel_z[i] -= 0;
				}
			}
		}

#pragma vector aligned
#pragma ivdep
		for (unsigned int i = 0; i < N + PADDING; ++i)
		{
			p->pos_x[i] += p->vel_x[i] * DT; //o
			p->pos_y[i] += p->vel_y[i] * DT; //o 
			p->pos_z[i] += p->vel_z[i] * DT; //o	
		}

	}
}


void particles_read(FILE *fp, t_particles *p)
{
    int i;
    for (i = 0; i < N; i++)
    {
        fscanf(fp, "%f %f %f %f %f %f %f \n",
            &(p->pos_x[i]), &(p->pos_y[i]), &(p->pos_z[i]),
            &(p->vel_x[i]), &(p->vel_y[i]), &(p->vel_z[i]),
            &(p->weight[i]));
		
    }

	for (i = N; i < N + PADDING; ++i)
	{
		p->weight[i] = 0;
		p->pos_x[i] = 0;
		p->pos_y[i] = 0;
		p->pos_z[i] = 0;
		p->vel_x[i] = 0;
		p->vel_y[i] = 0;
		p->vel_z[i] = 0;
	}


}

void particles_write(FILE *fp, t_particles *p)
{
    int i;
    for (i = 0; i < N; i++)
    {
        fprintf(fp, "%10.10f %10.10f %10.10f %10.10f %10.10f %10.10f %10.10f \n",
            p->pos_x[i], p->pos_y[i], p->pos_z[i],
            p->vel_x[i], p->vel_y[i], p->vel_z[i],
            p->weight[i]);
    }
}
