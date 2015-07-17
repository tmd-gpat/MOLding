/******************************************************************************
 The computer software and associated documentation called STAMP hereinafter
 referred to as the WORK which is more particularly identified and described in 
 the LICENSE.  Conditions and restrictions for use of
 this package are also in the LICENSE.

 The WORK is only available to licensed institutions.

 The WORK was developed by: 
	Robert B. Russell and Geoffrey J. Barton

 Of current contact addresses:

 Robert B. Russell (RBR)             Geoffrey J. Barton (GJB)
 Bioinformatics                      EMBL-European Bioinformatics Institute
 SmithKline Beecham Pharmaceuticals  Wellcome Trust Genome Campus
 New Frontiers Science Park (North)  Hinxton, Cambridge, CB10 1SD U.K.
 Harlow, Essex, CM19 5AW, U.K.       
 Tel: +44 1279 622 884               Tel: +44 1223 494 414
 FAX: +44 1279 622 200               FAX: +44 1223 494 468
 e-mail: russelr1@mh.uk.sbphrd.com   e-mail geoff@ebi.ac.uk
                                     WWW: http://barton.ebi.ac.uk/

   The WORK is Copyright (1997,1998,1999) Robert B. Russell & Geoffrey J. Barton
	
	
	

 All use of the WORK must cite: 
 R.B. Russell and G.J. Barton, "Multiple Protein Sequence Alignment From Tertiary
  Structure Comparison: Assignment of Global and Residue Confidence Levels",
  PROTEINS: Structure, Function, and Genetics, 14:309--323 (1992).
*****************************************************************************/
#include <stdio.h>
#include <math.h>
#include <stdlib.h>

#include <stamp.h>
#include <gjutil.h>
#include <gjnoc.h>
#define NOC_PARMS "noc sim single"

/******
 * treewise: Takes an upper diagonal matrix either derived from
 *  the pairwise comparisons, or some other method, generates
 *  a tree by single linkage cluster analysis (GJB's program OC) and 
 *  generates a multiple sequence alignment following the tree 
 *
 * RBR last modified 26 September 1995 
 *
 *  Arguments:
 *    struct domain_loc *domain -> array of domain descriptors
 *    long int ndomain -> number of domain descriptors
 *    struct parameters *parms -> STAMP parameters
 *  Return:
 *    int 0 for success, -1 for failure
 ******/

int treewise(struct domain_loc *domain, long int ndomain, 	
	     struct parameters *parms) {
  
  char *use;
  char noc_parms[200];
  char **ids;
  
  int i,j,k,l,m/*,n,nn,st*/;
/*  int ijunk,iter; */
/*  int pysize, pxsize; */
  int length/*,indx,indy*/;
  int nfit,testnum;
  int nclust;
/*  int **prob; */
  
  float fjunk;
/*  float diff; */
/*  float D,C,P; */
  float *Dij,*Pij,*dist,*Pijp;
  float rms,score/*,oldscore*/;
/*  float R2[3][3],V2[3]; */
  double **matrix;
  
  FILE /**IN,*OUT,*/*MAT;
  
  struct cluster *cl;
  
  Pijp=(float*)malloc(parms[0].MAX_SEQ_LEN*sizeof(float));
  Dij=(float*)malloc(parms[0].MAX_SEQ_LEN*sizeof(float));
  Pij=(float*)malloc(parms[0].MAX_SEQ_LEN*sizeof(float));
  dist=(float*)malloc(parms[0].MAX_SEQ_LEN*sizeof(float));
  for(i=0; i<parms[0].MAX_SEQ_LEN; ++i) 
    Pijp[i]=Dij[i]=Pij[i]=dist[i]=0.0;
  
  use=(char*)malloc(parms[0].MAX_SEQ_LEN*sizeof(char));
  
  if(ndomain<2) {
    fprintf(stderr,"error you can't run TREEWISE mode on one domain\n");
    fprintf(stderr,"  have you forgot -s ?\n");
    exit(-1);
  }
  
  fprintf(parms[0].LOG,"\n\nTREEWISE calculations\n\n");
  
  
  /* Read in the matrix file generated by the pairwise comparisons */
  printf("Reading in matrix file %s...\n",parms[0].matfile);
  if((MAT=fopen(parms[0].matfile,"r"))==NULL) {
    fprintf(stderr,"error opening file %s\n",parms[0].matfile);
    fprintf(stderr," have the pairwise comparisons been run?\n");
    exit(-1);
  }
  matrix=GJDudarr(ndomain);
  fscanf(MAT,"%d",&testnum);
  if(testnum!=ndomain) {
    fprintf(stderr,"error: matrix file %s contains %d elements\n",parms[0].matfile,testnum);
    fprintf(stderr,"  domain file contains %ld elements\n",ndomain);
    exit(-1);
  }
  
  for(i=0; i<ndomain; ++i) {
    for(j=i+1; j<ndomain; ++j)  {
      if(fscanf(MAT,"%f",&fjunk)==(char)EOF) {
	fprintf(stderr,"error: premature end of matrix file\n");
	exit(-1);
      }
      matrix[i][j-i-1]=(double)fjunk;
    }
  }
  fclose(MAT);
  printf("Doing cluster analysis...\n");
  ids=(char**)malloc(ndomain*sizeof(char*));
  for(i=0; i<ndomain; ++i) ids[i]=domain[i].id;
  strcpy(&noc_parms[0],NOC_PARMS);
  cl=get_clust(matrix,ids,ndomain,noc_parms);
  nclust=ndomain-1;
  
  
  
  for(i=0; i<(ndomain-1); ++i) {
    fprintf(parms[0].LOG,"cluster: %d\n",i+1);
    for(j=0; j<cl[i].a.number; ++j) 
      fprintf(parms[0].LOG,"%s ",domain[cl[i].a.member[j]].id);
    fprintf(parms[0].LOG,"\nand\n");
    for(j=0; j<cl[i].b.number; ++j) 
      fprintf(parms[0].LOG,"%s ",domain[cl[i].b.member[j]].id); 
    fprintf(parms[0].LOG,"\n\n");
  }
  
  for(i=0; i<ndomain; ++i) {
    /* also, the sequences are set equal to domain.aa initially */
    domain[i].align=(char*)malloc(parms[0].MAX_SEQ_LEN*sizeof(char));
    domain[i].oldalign=(char*)malloc(parms[0].MAX_SEQ_LEN*sizeof(char));
    for(j=0; j<domain[i].ncoords; ++j) {
      domain[i].align[j]=domain[i].aa[j];
      domain[i].oldalign[j]=domain[i].aa[j];
    }
    domain[i].align[j]=domain[i].oldalign[j]='\0';
  }
  
  /* In this bit, the clusters are considered one at a time and
   *  the coordinates fitted accordingly. */
  for(i=0; i<(ndomain-1); ++i) { 
    /* there are always ndomain-1 clusters */
    fprintf(parms[0].LOG,"Cluster: %d\n",(i+1));
    fprintf(parms[0].LOG,"Combining:\n");
    for(j=0; j<cl[i].a.number; ++j) fprintf(parms[0].LOG,"%s ",domain[cl[i].a.member[j]].id);
    fprintf(parms[0].LOG,"\nand\n");
    for(j=0; j<cl[i].b.number; ++j) fprintf(parms[0].LOG,"%s ",domain[cl[i].b.member[j]].id);
    fprintf(parms[0].LOG,"\n\n");
    
    /* set the transformations for the B cluster to the identity matrix and zero vector 
     * we do this since we are always leaving the coordinates transformed after each 
     *  treewise fit, and we want to save all of them together */
    for(j=0; j<cl[i].b.number; ++j) {
      k=cl[i].b.member[j];
      for(l=0; l<3; ++l) { 
	domain[k].v[l]=0.0; 
	for(m=0; m<3; ++m) {
	  if(m==l) domain[k].r[m][l]=1.0;
	  else domain[k].r[m][l]=0.0;
	}
      }
    }
    
    /*	   fprintf(parms[0].LOG,"Cluster B: \n"); 
	   for(j=0; j<cl[i].b.number; ++j) {
	   fprintf(parms[0].LOG,"%s\n",domain[cl[i].b.member[j]].id);
	   disp(domain[cl[i].b.member[j]],parms[0].LOG);
	   }
    */
    if(parms[0].NPASS==2) {
      if(parms[0].BOOLEAN) 
	fprintf(parms[0].LOG,"First fit: BOOLCUT = %5.3f\n",parms[0].first_BOOLCUT);
      else
	fprintf(parms[0].LOG,"First fit: E1 = %5.2f, E2 = %5.2f, CUT=%5.2f, PEN=%5.2f,TREEPEN=%5.2f\n",
		parms[0].first_E1,parms[0].first_E2,parms[0].first_CUTOFF,parms[0].first_PAIRPEN,parms[0].first_TREEPEN);
      parms[0].const1=-2*parms[0].first_E1*parms[0].first_E1;
      parms[0].const2=-2*parms[0].first_E2*parms[0].first_E2;
      parms[0].CUTOFF=parms[0].first_CUTOFF;
      parms[0].PAIRPEN=parms[0].first_PAIRPEN;
      parms[0].TREEPEN=parms[0].first_TREEPEN;
      parms[0].BOOLCUT=parms[0].first_BOOLCUT;
      if(treefit(domain,ndomain,cl[i],&score,&rms,&length,&nfit,Pij,Dij,dist,Pijp,1,0,parms)==-1) return -1;
      fprintf(parms[0].LOG,"Second fit: ");
    } else fprintf(parms[0].LOG,"Fitting with: ");
    if(parms[0].BOOLEAN) 
      fprintf(parms[0].LOG,"BOOLCUT = %5.3f\n",parms[0].second_BOOLCUT);
    else
      fprintf(parms[0].LOG,"E1 = %5.2f, E2 = %5.2f, CUT=%5.2f, PEN=%5.2f,TREEPEN=%5.2f\n",
	      parms[0].second_E1,parms[0].second_E2,parms[0].second_CUTOFF,parms[0].second_PAIRPEN,parms[0].second_TREEPEN);
    parms[0].const1=-2*parms[0].second_E1*parms[0].second_E1;
    parms[0].const2=-2*parms[0].second_E2*parms[0].second_E2;
    parms[0].CUTOFF=parms[0].second_CUTOFF;
    parms[0].PAIRPEN=parms[0].second_PAIRPEN;
    parms[0].TREEPEN=parms[0].second_TREEPEN;
    parms[0].BOOLCUT=parms[0].second_BOOLCUT;
    if(treefit(domain,ndomain,cl[i],&score,&rms,&length,&nfit,Pij,Dij,dist,Pijp,0,parms[0].TREEALIGN,parms)==-1) return -1;
    if(strcmp(parms[0].logfile,"silent")!=0) {
      fprintf(parms[0].LOG,"Sum: cluster: %3d, Sc: %7.3f, RMS: %7.3f, Len: %d, nfit: %d\n\n",
	      i+1,score,rms,length,nfit);
    } else {
      printf("Cluster: %2d (",i+1);
      for(j=0; j<cl[i].a.number; ++j) printf("%8s ",domain[cl[i].a.member[j]].id);
      printf(" & ");
      for(j=0; j<cl[i].b.number; ++j) printf("%8s ",domain[cl[i].b.member[j]].id);
      printf(") Sc %5.2f RMS %6.2f Len %3d nfit %3d ", score,rms,length,nfit);
      if(score<2.0) {
	printf(" LOW SCORE ");
      }
      printf("\n");
      printf(" See file %s.%d for the alignment and transformations\n",parms[0].transprefix,i+1);
    }
    /* updating the original transformation according the the most recent fit */
    for(j=0; j<cl[i].b.number; ++j) {
      k=cl[i].b.member[j];
      update(domain[k].r,domain[k].R,domain[k].v,domain[k].V);
    }
    /* outputing the results */
    if(makefile(domain,ndomain,cl[i],i,score,rms,length,nfit,Pij,Dij,dist,Pijp,0,parms)==-1) return -1;
  } 
  
  /* Various freeing */
  free(Pij); free(Dij); 
  free(Pijp); free(dist); 
  free(use); 
  
  for(i=0; i<ndomain-1; ++i) {
    free(cl[i].a.member);
    free(cl[i].b.member);
  }
  free(cl);
  
  for(i=0; i<ndomain; ++i) {
    free(domain[i].align);
    free(domain[i].oldalign);
  }
  free(ids);
  return 0;
}