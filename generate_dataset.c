/*
 * Headless dataset generator matching the update logic in simulation.c.
 *
 * Preserved from simulation.c:
 *   rod_1=150, rod_2=150, m1=10, m2=2, g=1
 *   initial state: a1=pi/2, a2=pi/2, v1=0, v2=0
 *   dt=0.01
 *   exactly 16 semi-implicit Euler updates per visual frame
 *   same acceleration equations and same solve() threshold (1e-12)
 *
 * No RK4, random initial conditions, or substituted timestep.
 *
 * Build:
 *   gcc -O3 -Wall -Wextra -pedantic generate_dataset_exact.c -lm -o generate_dataset
 *
 * Run:
 *   ./generate_dataset [number_of_visual_frames] [output.csv]
 *
 * Example:
 *   ./generate_dataset 2000 double_pendulum.csv
 *
 * Each visual frame contributes 16 samples, one for each inner update.
 */

#include <errno.h>
#include <inttypes.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* Exact constants from simulation.c. */
#define rod_1 150
#define rod_2 150
#define m1 10
#define m2 2
#define g 1
#define pi M_PI

/* State variables corresponding to simulation.c. */
static double v1, v2;
static double a1, a2;
static double acc1, acc2;

/* Same 2x2 solver as simulation.c. */
static int solve(double A, double B, double C,
                 double D, double E, double F,
                 double *x, double *y)
{
    double determinant = A * E - D * B;

    if (fabs(determinant) < 1e-12)
        return 0;

    *x = (C * E - F * B) / determinant;
    *y = (A * F - D * C) / determinant;

    return 1;
}

/* Same initial values as init() in simulation.c. */
static void init(void)
{
    acc1 = 0;
    acc2 = 0;
    v1 = 0;
    v2 = 0;
    a1 = pi / 2;
    a2 = pi / 2;
}

/*
 * Same dt and same 16-iteration loop as calculate_pos().
 * One call represents one visual frame in simulation.c.
 */
static int calculate_pos(FILE *out, uint64_t frame, uint64_t *step)
{
    double dt = 0.01;

    for (int i = 0; i < 16; i++)
    {
        double c1 = m2 * rod_2 * v2 * v2 * sinf(a2-a1)
                  - (m1+m2) * g * sin(a1);
        double d1 = (m1+m2) * rod_1;
        double b1 = m2 * rod_2 * cosf(a2-a1);

        double c2 = -rod_1 * v1 * v1 * sinf(a2-a1)
                  - g * sinf(a2);
        double d2 = rod_1 * cosf(a2-a1);
        double b2 = rod_2;

        /* Same solver call. If it fails, acc1/acc2 retain their old values. */
        solve(d1, b1, c1, d2, b2, c2, &acc1, &acc2);

        /*
         * Record the current state and the acceleration calculated from it,
         * before applying the update. The model learns state -> acceleration.
         */
        if (fprintf(out,
                    "0,%.10f,%.17g,%.17g,%.17g,%.17g,"
                    "%d,%d,%d,%d,%d,%.17g,%.17g,%" PRIu64 ",%d\n",
                    (double)(*step) * dt,
                    a1, a2, v1, v2,
                    m1, m2, rod_1, rod_2, g,
                    acc1, acc2, frame, i) < 0)
        {
            return 0;
        }

        /* Exact update order from simulation.c: velocities, then angles. */
        v1 += acc1 * dt;
        v2 += acc2 * dt;

        a1 += v1 * dt;
        a2 += v2 * dt;

        (*step)++;
    }

    return 1;
}

static int parse_frames(const char *text, uint64_t *frames)
{
    char *end = NULL;
    unsigned long long value;

    errno = 0;
    value = strtoull(text, &end, 10);

    if (errno != 0 || end == text || *end != '\0' || value == 0)
    {
        fprintf(stderr, "Invalid frame count: %s\n", text);
        return 0;
    }

    *frames = (uint64_t)value;
    return 1;
}

int main(int argc, char **argv)
{
    uint64_t frames = 2000;
    uint64_t step = 0;
    const char *output_path = "double_pendulum.csv";
    FILE *out;

    if (argc > 3)
    {
        fprintf(stderr, "Usage: %s [number_of_visual_frames] [output.csv]\n",
                argv[0]);
        return EXIT_FAILURE;
    }

    if (argc >= 2 && !parse_frames(argv[1], &frames))
        return EXIT_FAILURE;

    if (argc >= 3)
        output_path = argv[2];

    init();

    out = fopen(output_path, "w");
    if (out == NULL)
    {
        perror(output_path);
        return EXIT_FAILURE;
    }

    if (fprintf(out,
                "trajectory_id,time_s,theta1_rad,theta2_rad,"
                "omega1_rad_s,omega2_rad_s,m1,m2,L1,L2,g,"
                "alpha1_rad_s2,alpha2_rad_s2,frame,substep\n") < 0)
    {
        perror("Writing CSV header");
        fclose(out);
        return EXIT_FAILURE;
    }

    fprintf(stderr,
            "Generating %" PRIu64 " visual frames.\n"
            "dt=0.01; exactly 16 updates per frame.\n"
            "Initial state: a1=pi/2, a2=pi/2, v1=0, v2=0.\n",
            frames);

    for (uint64_t frame = 0; frame < frames; ++frame)
    {
        if (!calculate_pos(out, frame, &step))
        {
            fprintf(stderr, "CSV write failed at frame %" PRIu64 ".\n", frame);
            fclose(out);
            return EXIT_FAILURE;
        }

        if ((frame + 1) % 100 == 0 || frame + 1 == frames)
        {
            fprintf(stderr, "Completed %" PRIu64 "/%" PRIu64 " frames.\n",
                    frame + 1, frames);
        }
    }

    if (fclose(out) != 0)
    {
        perror("Closing output file");
        return EXIT_FAILURE;
    }

    printf("Dataset written to: %s\n", output_path);
    printf("Rows: %" PRIu64 "\n", step);
    printf("Simulated time represented: %.2f seconds\n", (double)step * 0.01);
    printf("Each visual frame represents 16 * 0.01 = 0.16 simulated seconds.\n");

    return EXIT_SUCCESS;
}
