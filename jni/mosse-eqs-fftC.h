#include <gsl/gsl_matrix.h> /* for matrix exponential*/

/*
   In this more simple version of the integrator, there is just one
   structure: 
     "mosse_fft"
   This holds everything needed to perform an integration.  It will
   probably be the case that we will want to make many of these to do
   things efficiently, but the C code is agnostic about this, and will
   just provide a few functions to get data in and out of these, as
   well as perform the integrations.

   This version allows there to be multiple nested plans.
*/
typedef struct {
  int n_fft; /* number of different plans */

  int nx;    /* x-extent */
  double dx; /* distance between x's */

  int *nd;    /* number of dimensions for each plan */
  int max_nd; /* max nd */

  /* Data */
  double *x;
  fftw_complex *y;

  /* Diversification information */
  double *lambda;
  double *mu;
  double *r;     /* for matrix exponential: substitution rate per bin (length ndat) */
  double a;      /* for matrix exponential: punctuational amplitude (scalar; a >= 0) */
  gsl_matrix *Q; /* for matrix exponential: 4x4 substitution-rate matrix */

  /* Drift and diffusion parameters */
  double drift;
  double diffusion;
  double dt;     /* for matrix exponential: time step (so propagate_t_mosse can read it) */

  /* Scratch space */
  double *z; /* Generally stores exp(-rt) */
  double *zz; /*for matrix exponential*/
  double *wrk;
  double *wrkd;
  gsl_matrix *Qx; /*for matrix exponential*/
  gsl_matrix *eQ; /*for matrix exponential*/

  /* Transform for x propagation */
  rfftw_plan_real **fft;

  /* Kernel information (space propagation) */
  double ny;      /* Fourier space extent */
  fftw_complex *fkern; /* Gaussian kernel transformation */
  int nkl;        /* Kernel width to the left */
  int nkr;        /* Kernel width to the right */
  int npad;       /* nkl + nkr + 1 */
  int ndat;       /* nx - npad */

  /* The kernel itself */
  double  *kern_x;
  fftw_complex *kern_y;
  rfftw_plan_real *kernel;
} mosse_fft;

mosse_fft* make_mosse_fft(int n_fft, int nx, double dx, int *nd, 
			    int flags);
void qf_copy_x_mosse(mosse_fft *obj, double *x, int nd, int copy_in);
void qf_copy_ED_mosse(mosse_fft *obj, double *x, int idx);
void qf_setup_kern_mosse(mosse_fft *obj, double drift, double diffusion,
		   double dt, int nkl, int nkr);
void do_integrate_mosse(mosse_fft *obj, int nt, int idx);
void propagate_t_mosse(mosse_fft *obj, int idx);
void propagate_x_mosse(mosse_fft *obj, int idx);


