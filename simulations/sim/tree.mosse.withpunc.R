## to simulate tree:
## pars is a list of:
## lambda, mu are speciation and extinction functions and return the speciation and extinction rates, given a vector of character states 'x'.
## char is a function describing evolution in substitution rate on log scale, randomly chosing a new rate given a vector of current rates 'x' and a period of time 'dt' over which evolution occurs.
## x0 is the root substitution rate on log scale
## max.taxa, max.t stop the simulation when reaching maximum number of taxa or maximum amount of time
## single.lineage if root is a single lineage
## to simulate sequence:
## l is the length of each category, k is the number of rate categories among sites, so l*k is the total sequence length
## Q substitution matrix
## bf root base frequency
## alpha shape parameter for gamma distribution of rate variation among sites, the distribution is divided into k categories
## to simulate tip trait
## using a regression function of simulated tip substitution rate ~ x0 + trait value * beta + rnorm(1, 0,sig), when beta=0, simulated tip substitution rate are normal distribution with mean x0 and sd sig.

require(phangorn)
require(LaplacesDemon)
require(diversitree)

tree.mosse <- function(pars, max.taxa=Inf, n.taxa=NULL, gsa=T, max.t=Inf, x0,
                        single.lineage=TRUE, l, Q, PP, bf, alpha, k, beta, sig, a, treefile, seqfile, traitfile) {
  if ( is.na(x0) )
    stop("x0 must be specified")
  else if ( length(x0) != 1 )
    stop("x0 must be of length 1")
  stopifnot(is.list(pars), all(sapply(pars, is.function)))

  out <- make.tree.mosse(pars, max.taxa, max.t, x0, single.lineage)
  info <- out[[1]]
  info2 <- out[[2]]                         
                         
  if ( single.lineage ) {
  	info2$time <- info2$time-info$time[1]
  	info$time <- info$time-info$time[1]
    info <- info[-1,]
  }
  phy <- diversitree:::me.to.ape.quasse(info)
  
  if (gsa) {
  	a <- ltt.plot.coords(phy, backward=F)
  	tmp <- which(a[,2]==n.taxa)
  	periods <- cbind(a[tmp-1,1],a[tmp,1],a[tmp,1]-a[tmp-1,1]) #each row is a time period (starting, ending, duration) with n.taxa
  	chosen.period <- sample.int(dim(periods)[1],size=1,prob=periods[,3])
  	chosen.time <- runif(1,min=periods[chosen.period,1],max=periods[chosen.period,2])
  	info.tmp <- info[info$time<=chosen.time,]
  	info2 <- info2[info2$time<=chosen.time,]
  	child <- setdiff(info$idx[rep(match(union(info.tmp$parent,info.tmp$idx[info.tmp$split]),info$parent),each=2)+c(0,1)],info.tmp$idx)
  	tmp <- sapply(child, function (i) max(which(info2$idx==i)))
  	if(any(is.infinite(tmp))) {
  		child <- child[is.infinite(tmp)]
  		tmp <- tmp[is.finite(tmp)]
  		info.tmp2 <- info[info$idx==child,]
  		info.tmp2$len <- info.tmp2$len-(info.tmp2$time-chosen.time)
  		info.tmp2$time <- chosen.time
  		info.tmp2$state <- info.tmp$state[match(info.tmp2$parent,info.tmp$idx)]
  		info.tmp2$split <- FALSE
  		info2 <- rbind(info2,info.tmp2)
  		info.tmp <- rbind(info.tmp,info.tmp2)
  	}
  	rownames(info.tmp) <- paste0("X",rownames(info.tmp))
  	info <- rbind(info.tmp,info2[tmp,])
  	info <- info[order(info$idx),]
  	rownames(info) <- info$idx
  	phy <- diversitree:::me.to.ape.quasse(info)
  }
  
  names(phy$edge.state) <- phy$edge[,2]
  
  phy <- prune.mosse(phy)
  
  seq <- simSeq.mosse(phy, l, bf, Q, PP, alpha, k, info2)
  
  eps <- rnorm(length(phy$tip.label),0,sig)
  if (beta!=0) {
  	trait <- (phy$tip.state - x0 - eps) / beta
  } else {
  	#trait <- rep(x0,length(phy$tip.label))
  	trait <- rnorm(length(phy$tip.label),0,1)
  }
  trait <- exp(trait)
  names(trait) <- phy$tip.label
  
  write.phyDat(seq,seqfile,format="fasta",nbcol=-1,colsep="")
  write.tree(phy,treefile)
  write.csv(trait, traitfile)
  
  list(phy,seq, trait)
}


make.tree.mosse <- function(pars, max.taxa=Inf, max.t=Inf, x0,
                             single.lineage=TRUE, kk=50, ...) {
  lambda <- pars[[1]]
  mu     <- pars[[2]]
  char   <- pars[[3]]
  
  ## Always assuming *two* lineages for now, both in state x0.  I am
  ## adding a very small amount of time here so that the root does not
  ## have a real zero branch length (this happens 1/k of the time, and
  ## breaks quassecache).
  if ( single.lineage ) {
    info <- data.frame(idx=1, len=1e-8, parent=0, state=x0,
                       extinct=FALSE, split=FALSE, time=0)
  } else {
    info <- data.frame(idx=1:2, len=1e-8, parent=0, state=x0,
                       extinct=FALSE, split=FALSE, time=0)
  }
   
  info2 <- info 
  lineages <- which(!info$extinct & !info$split)
  n.taxa <- length(lineages)
  t <- 0
  t.left <- max.t

  while ( n.taxa <= max.taxa && n.taxa > 0 && t.left > 0 ) {
    x <- run.until.change(lineages, info, info2, k, lambda, mu, char, t.left, t)
    lineages <- x[[1]]
    info <- x[[2]]
    n.taxa <- length(lineages)
    t <- t + x[[4]]
    t.left <- t.left - x[[4]]
    info2 <- x[[5]]
  }

  if ( n.taxa > max.taxa ) {
    ## Drop final speciation event.
    drop <- info[nrow(info)-1,]
    info$split[drop$parent] <- FALSE
    info$state[drop$parent] <- drop$state
    info$len[drop$parent] <- info$len[drop$parent] + drop$len
    info$time[drop$parent] <- t
    info <- info[seq_len(nrow(info)-2),]
    
    idx <- max(which(info2$idx==drop$parent))
    info2[idx,] <- info[drop$parent,]
  }

  attr(info, "t") <- t
  list(info,info2)
}

## This function extends lineages until either a speciation or
## extinction event happens.  The time interval is rescaled so that
## 'k' character change events are expected between each speciation
## and extinction event.
## TODO: I am not truncating exactly the time interval when we might
## run into the maximum time.  If 'k' is sufficiently high, this
## should not matter much.
run.until.change <- function(lineages, info, info2, kk, lambda, mu, char,
                             t.left, t) {
  i <- 1
  time <- 0
  n.extant <- length(lineages)
  p.change <- 1/kk
  niter <- 1
  repeat {
    state <- info$state[lineages]
    lx <- lambda(state)
    mx <- mu(state)
    r <- sum(lx + mx)
    dt <- 1/(r*kk)
    
    if ( runif(1) < p.change ) {
      if ( runif(1) < sum(lx)/r ) { # speciation
        i <- sample(n.extant, 1, prob=lx)
        info <- speciate(info, lineages[i], t)
        lineages <- c(lineages[-i], c(-1,0) + nrow(info))
      } else {
        i <- sample(n.extant, 1, prob=mx)
        info$extinct[lineages[i]] <- TRUE
        lineages <- lineages[-i]
      }
      info$len[lineages]   <- info$len[lineages] + dt
      info$state[lineages] <- char(info$state[lineages], dt)
      info$time[lineages]   <- info$time[lineages] + dt
      info2 <- rbind(info2,info[lineages,])
      time <- time + dt
      t <- t + dt
      break
    }

    info$len[lineages]   <- info$len[lineages] + dt
    info$state[lineages] <- char(info$state[lineages], dt)
    info$time[lineages]   <- info$time[lineages] + dt
    info2 <- rbind(info2,info[lineages,])
    niter <- niter + 1
    time <- time + dt
    t <- t + dt

    if ( time > t.left )
      break
  }

  list(lineages, info, length(lineages) - n.extant, time, info2)
}

## This function rejigs the 'lineages' structure when speciation
## happens, creating new species.
speciate <- function(info, i, t) {
  j <- 1:2 + nrow(info)
  info[j,"idx"] <- j
  info[j,-1] <- list(len=0, parent=i,
                     state=info$state[i],
                     extinct=FALSE, split=FALSE, time=t)
  info$split[i] <- TRUE
  info
}
