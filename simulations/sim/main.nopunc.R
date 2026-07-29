library(diversitree)
library(phangorn)
library(LaplacesDemon)

# simulation based on parameter values for birds

#speciation rate is 0.034-2.25 (Maliet)
#mitochondrial substitution rate is 0.0035~0.0566 for third codon, or 0.001~0.0158 for all codon positions (Nabholz)
#median substitution rate is 0.015 for third codon, or 0.0040 for all codon (Nabholz)
#speciation rate = 30*substitution rate (Lanfear, total branch length)
#average extinction rate = 0.0046
#drift=0
#diffusion=0.001

#define speciation rate x as a function of substitution rate r
lambda_r   <- 30
lambda_cap <- 3
linear.x <- function(x,r) {
	y <- r*exp(x)
	#y[x<=0] <- 0
	#y[x>0] <- r*x[x>0]
	y[y>lambda_cap] <- lambda_cap
	y
}
lambda <- function (x) linear.x(x, r=lambda_r)

#define extinction rate
mu_val <- 0.05
mu <- function (x) constant.x(x, mu_val)

#defind BW of substitution rate
# original value: drift(0,0.005)
drift_mean  <- -0.2
drift_sigma <- 0.2
char <- make.brownian.with.drift(drift_mean, drift_sigma)

#bundle parameters
pars <- c(lambda, mu, char)

#define root substitution rate
x0 <- 0.02

#define simulation parameters of the tree
max.taxa <- 100
n.taxa <- 20 #75

#define substitution model
l <- 500 #number of sites per gamma gategory (was 1000 in sim6; halved here for sim7_2k -> 4*500=2000 sites)
kappa <- rexp(1, 0.15)
Q <- c(1, kappa, 1, 1, kappa, 1) #HKY
bf <- as.vector(rdirichlet(1, alpha=c(1,1,1,1))) #base frequency
#alpha <- 20 #shape parameter for gamma category
alpha <- Inf
k <- 4 #number of gamma category

#spike; set to 0 if no spike
QQ <- matrix(0,4,4)
QQ[lower.tri(QQ)] <- Q
QQ[upper.tri(QQ)] <- t(QQ)[upper.tri(QQ)]
QQ[,1] <- QQ[,1]*bf[1]
QQ[,2] <- QQ[,2]*bf[2]
QQ[,3] <- QQ[,3]*bf[3]
QQ[,4] <- QQ[,4]*bf[4]
QQ[1,1] <- -sum(QQ[1,])
QQ[2,2] <- -sum(QQ[2,])
QQ[3,3] <- -sum(QQ[3,])
QQ[4,4] <- -sum(QQ[4,])
max_a <- 1/abs(min(diag(QQ))) #max a to make diag in PP non-negative
a <- 0 # no punctuation
# a <- runif(1,0,max_a)
# a= 0.0303049
PP <- diag(4) + a * QQ

#define tip regression model
beta <- 1 #regression coefficient in tipGLM
sig <- 0.2 #error term in tipGLM

#calculate bound for min and max bins, x is tip trait value
#x <- read.csv("~/trait1_0.001_8.csv",header=T,row.names=1)
#w <- 5
#betamin <- -3 #give beta prior N(0,1)
#betamax <- 3
#rminmin <- x0-w/2*betamin*max(x)+(w/2+1)*betamin*min(x)
#rminmax <- x0-w/2*betamax*max(x)+(w/2+1)*betamax*min(x)
#rmaxmin <- x0+(w/2+1)*betamin*max(x)-w/2*betamin*min(x)
#rmaxmax <- x0+(w/2+1)*betamax*max(x)-w/2*betamax*min(x)
#rmin <- min(rminmin,rminmax,rmaxmin,rmaxmax)
#rmax <- max(rminmin,rminmax,rmaxmin,rmaxmax)
#calculate bin size
#dx <- (rmax-rmin)/1024/4

#initiation simulation
sim <- vector("list",10)
seed <- round(runif(10,1,100))
for (i in 1:10) {
set.seed(seed[i])
#define results filename
treefile <- paste0("tree",beta,"_",sig,"_",seed[i],".tre") #tree file
seqfile <- paste0("seq",beta,"_",sig,"_",seed[i],".fasta") #sequence file
traitfile <- paste0("trait",beta,"_",sig,"_",seed[i],".csv") #tip trait file
sim[[i]] <- try(tree.mosse(pars=pars,max.taxa=max.taxa,n.taxa=n.taxa,gsa=T,x0=log(x0),l=l,Q=Q,PP=PP,bf=bf,alpha=alpha,k=k,beta=beta,sig=sig,a=a,treefile=treefile, seqfile=seqfile, traitfile=traitfile),silent=T)
while (class(sim[[i]])=="try-error") { #remove extinct tree or traits exceed upper substitution rate too much
	sim[[i]] <- try(tree.mosse(pars=pars,max.taxa=max.taxa,n.taxa=n.taxa,gsa=T,x0=log(x0),l=l,Q=Q,PP=PP, bf=bf,alpha=alpha,k=k,beta=beta,sig=sig,a=a,treefile=treefile, seqfile=seqfile, traitfile=traitfile),silent=T)
}
}

# --- write sim_truth.csv: one row per dataset, with the actual values used ---
suppressPackageStartupMessages(library(ape))
truth <- data.frame()
for (i in 1:10) {
  tf <- paste0("tree", beta, "_", sig, "_", seed[i], ".tre")
  tr <- read.tree(tf)
  root_h <- max(node.depth.edgelength(tr)[1:length(tr$tip.label)])
  truth <- rbind(truth, data.frame(
    seed         = seed[i],
    kappa        = kappa,
    bf_A         = bf[1],  bf_C = bf[2],  bf_G = bf[3],  bf_T = bf[4],
    x0_linear    = x0,
    x0_log       = log(x0),
    drift_mean   = drift_mean,
    drift_sigma  = drift_sigma,
    mu           = mu_val,
    lambda_r     = lambda_r,
    lambda_cap   = lambda_cap,
    punc_a       = a,
    beta         = beta,
    sig          = sig,
    n_taxa       = length(tr$tip.label),
    root_height  = root_h,
    n_sites      = l * k,
    alpha_gamma  = alpha
  ))
}
write.csv(truth, "sim_truth.csv", row.names = FALSE)
cat("wrote sim_truth.csv with", nrow(truth), "rows\n")

