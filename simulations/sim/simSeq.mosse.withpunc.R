simSeq.mosse <- function (phy,l,bf,Q,PP,alpha,k,info2,a=0) {
  if (alpha==Inf) {
  	rates <- rep(1,k)
  } else {
  	rates <- discrete.gamma(alpha, k)
  }	
  levels <- c("a", "c", "g", "t")
  if (is.matrix(Q)) Q <- Q[lower.tri(Q)]
  eig <- phangorn:::edQt(Q, bf)

  m <- length(levels)

  rootseq <- sample(levels, l*k, replace = TRUE, prob = bf)
  
  edge <- phy$orig[phy$keep.idx,c("idx","parent")]
  nNodes <- nrow(edge)+1
  res <- matrix(NA, nNodes, l*k)
  parent <- as.character(edge[, 2])
  child <- as.character(edge[, 1])
  root <- parent[1]
  rownames(res) <- c(root,edge$idx)
  res[root, ] <- rootseq
  for (i in 1:length(parent)) {
    aa <- 1
    from <- parent[i]
    to <- child[i]
    hist <- info2[info2$idx==to,]
    nt <- nrow(hist)
    dt <- c(hist$len[1],hist$len[-1]-hist$len[-nt])
    r <- exp(hist$state)
    #coount hidden specitio event
    sis <- child[parent==from]
    while (length(sis)==1) {
        aa <- aa+1
        from2 <- sis
        sis <- child[parent==from2]
    }
    #r[r<0] <- 0
    for (j in 1:k) {
    	r <- r * rates[j]
    	P <- Reduce("%*%", lapply(1:nt, function (ii) getP.mosse(dt[ii], eig, r[ii])))
        while(aa>0) {
            P <- P %*% PP
            aa <- aa-1
        }
    	for (z in 1:m) {
    		idx <- ((j-1)*l+1):(j*l)
      		ind <- res[from, idx] == levels[z]
     		res[to, idx[ind]] <- sample(levels, sum(ind), replace = TRUE, prob = P[, z])
   		 }
    }
  }
  tip <- as.character(phy$orig[phy$keep.tip,"idx"])
  res <- res[tip,]
  rownames(res) <- phy$tip.label
  return(phangorn:::phyDat.DNA(res, return.index = TRUE))
}

getP.mosse <- function (t, eig, rate) {
	P <- phangorn:::getP(t, eig, rate)[[1]]
	# avoid numerical problems for larger P and small t
    if (any(P < 0)) P[P < 0] <- 0
    P
}
