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

  /* Eigendecomposition fast computation for exp(Q*scalar): Q = U * diag(eVal) * U^-1 */
  double *eVal;   /* eigenvalues -- length 4 */
  double *eVec;   /* eigenvectors -- length 16 */
  double *iEvec;  /* inverses of eigenvectors -- length 16 */
  int useEigen;   /* 0 = use GSL, 1 = use eigendecomposition */

  /* Per-bin cache of eQ = exp(Q * r[ix] * dt) for the a==0. When
   * a == 0 the per-bin exponent is invariant across the dt-step loop, so the
   * matrix is built once per likelihood call and reused. Length nx*16
   * (row-major: eQ_cache[ix*16 + i*4 + j]). eQ_cache_valid signals whether
   * the contents are current (set in JNI doIntegrateMosse, cleared otherwise). */
  double *eQ_cache;     /* length nx*16 */
  int    eQ_cache_valid; /* 0 = recompute per step (a > 0 or GSL path), 1 = use cache */

  /* Generation counter for eigendata (eVal/eVec/iEvec/eQ_cache). Java passes
   * a monotonically-increasing generation that bumps whenever the substitution
   * model's eigendecomposition or eQCache is rebuilt (i.e. whenever
   * transitionMatricesDirty fires). When the passed-in generation matches
   * obj->eigen_generation, the four per-call array copies are skipped — these
   * arrays are constant across all branch calls within a single likelihood eval,
   * so ~4 (eVal/eVec/iEvec) + ~520 KB (eQ_cache) of memcpy/safepoint cost per
   * branch is avoided after the first call from each thread. Initialised to -1
   * in make_mosse_fft so the very first call (Java gen 0 or 1) always copies. */
  long eigen_generation;

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

  double *xt; /* transposed D-values buffer for propagate_t */

  /* The kernel itself */
  double  *kern_x;
  fftw_complex *kern_y;
  rfftw_plan_real *kernel;

  /* Per-plan reusable buffers for arrays marshalled from Java each call.
   * Pre-allocated once in make_mosse_fft and reused for every
   * doIntegrateMosse to avoid per-call malloc/free/memcpy churn.
   * Sized to fit the maximum payload that can arrive for this plan:
   *   - lambda_buf, mu_buf, r_buf:  length nx  (numEntries <= nx)
   *   - vars_buf:                   length nx * max_nd
   */
  double *lambda_buf;
  double *mu_buf;
  double *r_buf;
  double *vars_buf;

  /* Cached kernel-setup inputs. qf_setup_kern_mosse becomes a no-op when
   * (drift, diffusion, dt, nkl, nkr) match the cached values, which saves
   * an FFT and an O(nx) loop on every branch call where these are constant
   * for the entire likelihood evaluation. */
  int    kern_valid;        /* 0 = must (re)build kernel, 1 = cached values match */
  double kern_drift;
  double kern_diffusion;
  double kern_dt;
  int    kern_nkl;
  int    kern_nkr;
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


