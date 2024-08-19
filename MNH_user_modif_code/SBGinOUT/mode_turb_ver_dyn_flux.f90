!MNH_LIC Copyright 1994-2021 CNRS, Meteo-France and Universite Paul Sabatier
!MNH_LIC This is part of the Meso-NH software governed by the CeCILL-C licence
!MNH_LIC version 1. See LICENSE, CeCILL-C_V1-en.txt and CeCILL-C_V1-fr.txt
!MNH_LIC for details. version 1.
MODULE MODE_TURB_VER_DYN_FLUX
IMPLICIT NONE
CONTAINS
SUBROUTINE TURB_VER_DYN_FLUX(D,CST,CSTURB,TURBN,TLES,KSV,O2D,OFLAT, &
                      KRR,OOCEAN,OCOUPLES,                          &
                      PEXPL,PTSTEP,TPFILE,                          &
                      PDXX,PDYY,PDZZ,PDZX,PDZY,PDIRCOSZW,PZZ,       &
                      PCOSSLOPE,PSINSLOPE,                          &
                      PRHODJ,                                       &
                      PSFUM,PSFVM,                                  &
                      PCDUEFF,PTAU11M,PTAU12M,PTAU33M,              &
                      PTHLM,PRM,PSVM,PUM,PVM,PWM,PUSLOPEM,PVSLOPEM, &
                      PTKEM,PLM,MFMOIST,PWU,PWV,                    &
                      PRUS,PRVS,PRWS,                               &
                      PDP,PTP,PSSUFL_C,PSSVFL_C,PSSUFL,PSSVFL       )
!     ###############################################################
!
!
!!****  *TURB_VER_DYN_FLUX* -compute the source terms due to the vertical turbulent
!!       fluxes.
!!
!!    PURPOSE
!!    -------
!       The purpose of this routine is to compute the vertical turbulent
!     fluxes of the evolutive variables and give back the source
!     terms to the main program.	In the case of large horizontal meshes,
!     the divergence of these vertical turbulent fluxes represent the whole
!     effect of the turbulence but when the three-dimensionnal version of
!     the turbulence scheme is activated (CTURBDIM="3DIM"), these divergences
!     are completed in the next routine TURB_HOR.
!		  An arbitrary degree of implicitness has been implemented for the
!     temporal treatment of these diffusion terms.
!       The vertical boundary conditions are as follows:
!           *  at the bottom, the surface fluxes are prescribed at the same
!              as the other turbulent fluxes
!           *  at the top, the turbulent fluxes are set to 0.
!       It should be noted that the condensation has been implicitely included
!     in this turbulence scheme by using conservative variables and computing
!     the subgrid variance of a statistical variable s indicating the presence
!     or not of condensation in a given mesh.
!
!!**  METHOD
!!    ------
!!      1D type calculations are made;
!!      The vertical turbulent fluxes are computed in an off-centered
!!      implicit scheme (a Crank-Nicholson type with coefficients different
!!      than 0.5), which allows to vary the degree of implicitness of the
!!      formulation.
!!      	 The different prognostic variables are treated one by one.
!!      The contributions of each turbulent fluxes are cumulated into the
!!      tendency  PRvarS, and into the dynamic and thermal production of
!!      TKE if necessary.
!!
!!			 In section 2 and 3, the thermodynamical fields are considered.
!!      Only the turbulent fluxes of the conservative variables
!!      (Thetal and Rnp stored in PRx(:,:,:,1))  are computed.
!!       Note that the turbulent fluxes at the vertical
!!      boundaries are given either by the soil scheme for the surface one
!!      ( at the same instant as the others fluxes) and equal to 0 at the
!!      top of the model. The thermal production is computed by vertically
!!      averaging the turbulent flux and multiply this flux at the mass point by
!!      a function ETHETA or EMOIST, which preform the transformation from the
!!      conservative variables to the virtual potential temperature.
!!
!! 	    In section 4, the variance of the statistical variable
!!      s indicating presence or not of condensation, is determined in function
!!      of the turbulent moments of the conservative variables and its
!!      squarred root is stored in PSIGS. This information will be completed in
!!      the horizontal turbulence if the turbulence dimensionality is not
!!      equal to "1DIM".
!!
!!			 In section 5, the x component of the stress tensor is computed.
!!      The surface flux <u'w'> is computed from the value of the surface
!!      fluxes computed in axes linked to the orography ( i", j" , k"):
!!        i" is parallel to the surface and in the direction of the maximum
!!           slope
!!        j" is also parallel to the surface and in the normal direction of
!!           the maximum slope
!!        k" is the normal to the surface
!!      In order to prevent numerical instability, the implicit scheme has
!!      been extended to the surface flux regarding to its dependence in
!!      function of U. The dependence in function of the other components
!!      introduced by the different rotations is only explicit.
!!      The turbulent fluxes are used to compute the dynamic production of
!!      TKE. For the last TKE level ( located at PDZZ(:,:,IKB)/2 from the
!!      ground), an harmonic extrapolation from the dynamic production at
!!      PDZZ(:,:,IKB) is used to avoid an evaluation of the gradient of U
!!      in the surface layer.
!!
!!         In section 6, the same steps are repeated but for the y direction
!!		  and in section 7, a diagnostic computation of the W variance is
!!      performed.
!!
!!         In section 8, the turbulent fluxes for the scalar variables are
!!      computed by the same way as the conservative thermodynamical variables
!!
!!
!!    EXTERNAL
!!    --------
!!      GX_U_M, GY_V_M, GZ_W_M :  cartesian gradient operators
!!      GX_U_UW,GY_V_VW	         (X,Y,Z) represent the direction of the gradient
!!                               _(M,U,...)_ represent the localization of the
!!                               field to be derivated
!!                               _(M,UW,...) represent the localization of the
!!                               field	derivated
!!
!!
!!      MXM,MXF,MYM,MYF,MZM,MZF
!!                             :  Shuman functions (mean operators)
!!      DXF,DYF,DZF,DZM
!!                             :  Shuman functions (difference operators)
!!
!!      SUBROUTINE TRIDIAG_WIND: to compute the split implicit evolution
!!                               of a variable located at a wind point
!!
!!      FUNCTIONs ETHETA and EMOIST  :
!!            allows to compute:
!!            - the coefficients for the turbulent correlation between
!!            any variable and the virtual potential temperature, of its
!!            correlations with the conservative potential temperature and
!!            the humidity conservative variable:
!!            -------              -------              -------
!!            A' Thv'  =  ETHETA   A' Thl'  +  EMOIST   A' Rnp'
!!
!!
!!    IMPLICIT ARGUMENTS
!!    ------------------
!!      Module MODD_CST : contains physical constants
!!
!!           CST%XG         : gravity constant
!!
!!      Module MODD_CTURB: contains the set of constants for
!!                        the turbulence scheme
!!
!!           CSTURB%XCMFS,XCMFB : cts for the momentum flux
!!           CSTURB%XCSHF       : ct for the sensible heat flux
!!           CSTURB%XCHF        : ct for the moisture flux
!!           CSTURB%XCTV,CSTURB%XCHV   : cts for the T and moisture variances
!!
!!      Module MODD_PARAMETERS
!!
!!           JPVEXT_TURB     : number of vertical external points
!!           JPHEXT     : number of horizontal external points
!!
!!
!!    REFERENCE
!!    ---------
!!      Book 1 of documentation (Chapter: Turbulence)
!!
!!    AUTHOR
!!    ------
!!      Joan Cuxart             * INM and Meteo-France *
!!
!!    MODIFICATIONS
!!    -------------
!!      Original       August   19, 1994
!!      Modifications: February 14, 1995 (J.Cuxart and J.Stein)
!!                                  Doctorization and Optimization
!!      Modifications: March 21, 1995 (J.M. Carriere)
!!                                  Introduction of cloud water
!!      Modifications: June  14, 1995 (J.Cuxart and J. Stein)
!!                                 Phi3 and Psi3 at w-point + bug in the all
!!                                 or nothing condens.
!!      Modifications: Sept  15, 1995 (J.Cuxart and J. Stein)
!!                                 Change the DP computation at the ground
!!      Modifications: October 10, 1995 (J.Cuxart and J. Stein)
!!                                 Psi for scal var and LES tools
!!      Modifications: November 10, 1995 (J. Stein)
!!                                 change the surface	relations
!!      Modifications: February 20, 1995 (J. Stein) optimization
!!      Modifications: May 21, 1996 (J. Stein)
!!                                  bug in the vertical flux of the V wind
!!                                  component for explicit computation
!!      Modifications: May 21, 1996 (N. wood)
!!                                  modify the computation of the vertical
!!                                   part or the surface tangential flux
!!      Modifications: May 21, 1996 (P. Jabouille)
!!                                  same modification in the Y direction
!!
!!      Modifications: Sept 17, 1996 (J. Stein) change the moist case by using
!!                                  Pi instead of Piref + use Atheta and Amoist
!!
!!      Modifications: Nov  24, 1997 (V. Masson) removes the DO loops
!!      Modifications: Mar  31, 1998 (V. Masson) splits the routine TURB_VER_DYN_FLUX
!!      Modifications: Oct  18, 2000 (J. Stein)  Bug in some computations for IKB level
!!      Modifications: Oct  18, 2000 (V. Masson) LES computations + OFLAT switch
!!                     Nov  06, 2002 (V. Masson) LES budgets
!!                     October 2009 (G. Tanguy) add ILENCH=LEN(YCOMMENT) after
!!                                              change of YCOMMENT
!!      2012-02 Y. Seity,  add possibility to run with reversed vertical levels
!!      Modifications  July 2015 (Wim de Rooy) TURBN%LHARATU switch
!!      J.Escobar : 15/09/2015 : WENO5 & JPHEXT <> 1
!!  Philippe Wautelet: 05/2016-04/2018: new data structures and calls for I/O
!!      Q. Rodier      17/01/2019 : cleaning : remove cyclic conditions on DP and ZA
!!      Modification   June 2019 (Wim de Rooy) 50*MF term can be removed with
!!                                             inclusion of energy cascade
!! JL Redelsperger 03/2021 : Add Ocean  & O-A Autocoupling LES Cases
!!--------------------------------------------------------------------------
!
!*      0. DECLARATIONS
!          ------------
!
USE PARKIND1,   ONLY: JPRB
USE MODE_SHUMAN_PHY
USE YOMHOOK,    ONLY: LHOOK, DR_HOOK
!
USE MODD_CST,            ONLY: CST_t
USE MODD_CTURB,          ONLY: CSTURB_t
USE MODD_DIMPHYEX,       ONLY: DIMPHYEX_t
USE MODD_FIELD,          ONLY: TFIELDMETADATA, TYPEREAL
USE MODD_IO,             ONLY: TFILEDATA
USE MODD_LES,            ONLY: TLES_t
USE MODD_PARAMETERS,     ONLY: JPVEXT_TURB, XUNDEF
USE MODD_TURB_n,         ONLY: TURB_t
!
USE MODE_GRADIENT_U_PHY, ONLY : GZ_U_UW_PHY, GX_U_M_PHY
USE MODE_GRADIENT_V_PHY, ONLY : GZ_V_VW_PHY, GY_V_M_PHY
USE MODE_GRADIENT_W_PHY, ONLY : GX_W_UW_PHY, GY_W_VW_PHY, GZ_W_M_PHY
USE MODE_GRADIENT_M_PHY, ONLY : GX_M_U_PHY, GY_M_V_PHY
USE MODE_IO_FIELD_WRITE_PHY, only: IO_FIELD_WRITE_PHY
USE MODE_ll
USE MODE_TRIDIAG_WIND,   ONLY: TRIDIAG_WIND
!
USE MODI_LES_MEAN_SUBGRID_PHY
USE MODI_SECOND_MNH
!
USE MODD_BLANK_n, ONLY: TDUMMY10,TDUMMY11  ! store VW_VFLX in OUT files, Hugo Jacquet 21/07/23

IMPLICIT NONE
!
!*      0.1  declarations of arguments
!
!
!
TYPE(DIMPHYEX_t),       INTENT(IN)   :: D
TYPE(CST_t),            INTENT(IN)   :: CST
TYPE(CSTURB_t),         INTENT(IN)   :: CSTURB
TYPE(TURB_t),           INTENT(IN)   :: TURBN
TYPE(TLES_t),           INTENT(INOUT):: TLES          ! modd_les structure
INTEGER,                INTENT(IN)   :: KSV           ! number of scalar variables
LOGICAL,                INTENT(IN)   ::  OOCEAN       ! switch for Ocean model version
LOGICAL,                INTENT(IN)   ::  O2D          ! Logical for 2D model version (modd_conf)
LOGICAL,                INTENT(IN)   ::  OFLAT        ! Logical for zero ororography
LOGICAL,                INTENT(IN)   ::  OCOUPLES     ! switch to activate atmos-ocean LES version 
INTEGER,                INTENT(IN)   ::  KRR          ! number of moist var.
REAL,                   INTENT(IN)   ::  PEXPL        ! Coef. for temporal disc.
REAL,                   INTENT(IN)   ::  PTSTEP       ! Double Time Step
TYPE(TFILEDATA),        INTENT(IN)   ::  TPFILE       ! Output file
!
REAL, DIMENSION(D%NIJT,D%NKT), INTENT(IN)   ::  PDXX, PDYY, PDZZ, PDZX, PDZY
                                                      ! Metric coefficients
REAL, DIMENSION(D%NIJT),   INTENT(IN)   ::  PDIRCOSZW    ! Director Cosinus of the
                                                      ! normal to the ground surface
REAL, DIMENSION(D%NIJT,D%NKT), INTENT(IN)   ::  PZZ          ! altitude of flux points
REAL, DIMENSION(D%NIJT),   INTENT(IN)   ::  PCOSSLOPE    ! cosinus of the angle
                                      ! between i and the slope vector
REAL, DIMENSION(D%NIJT),   INTENT(IN)   ::  PSINSLOPE    ! sinus of the angle
                                      ! between i and the slope vector
!
REAL, DIMENSION(D%NIJT,D%NKT), INTENT(IN)   ::  PRHODJ       ! dry density * grid volum
REAL, DIMENSION(D%NIJT,D%NKT), INTENT(IN)   ::  MFMOIST      ! moist mass flux dual scheme

!
REAL, DIMENSION(D%NIJT),   INTENT(IN)   ::  PSFUM,PSFVM !  normal momentum sfc flux
REAL, DIMENSION(D%NIJT),   INTENT(IN)   ::  PCDUEFF     ! Cd * || u || at time t
REAL, DIMENSION(D%NIJT),   INTENT(IN)   ::  PTAU11M      ! <uu> in the axes linked
       ! to the maximum slope direction and the surface normal and the binormal
       ! at time t - dt
REAL, DIMENSION(D%NIJT),   INTENT(IN)   ::  PTAU12M      ! <uv> in the same axes
REAL, DIMENSION(D%NIJT),   INTENT(IN)   ::  PTAU33M      ! <ww> in the same axes
!
REAL, DIMENSION(D%NIJT,D%NKT), INTENT(IN)   ::  PUM,PVM,PWM, PTHLM
  ! Wind at t-Delta t
REAL, DIMENSION(D%NIJT,D%NKT,KRR), INTENT(IN) ::  PRM
REAL, DIMENSION(D%NIJT,D%NKT,KSV), INTENT(IN) ::  PSVM
REAL, DIMENSION(D%NIJT),   INTENT(IN)   ::  PUSLOPEM     ! wind component along the
                                     ! maximum slope direction
REAL, DIMENSION(D%NIJT),   INTENT(IN)   ::  PVSLOPEM     ! wind component along the
                                     ! direction normal to the maximum slope one
!
REAL, DIMENSION(D%NIJT,D%NKT), INTENT(IN)   ::  PTKEM        ! TKE at time t
REAL, DIMENSION(D%NIJT,D%NKT), INTENT(IN)   ::  PLM          ! Turb. mixing length
REAL, DIMENSION(D%NIJT,D%NKT), INTENT(OUT)  ::  PWU          ! momentum flux u'w'
REAL, DIMENSION(D%NIJT,D%NKT), INTENT(OUT)  ::  PWV          ! momentum flux v'w'
!
REAL, DIMENSION(D%NIJT,D%NKT), INTENT(INOUT)   ::  PRUS, PRVS, PRWS
                            ! cumulated sources for the prognostic variables
!
REAL, DIMENSION(D%NIJT,D%NKT), INTENT(OUT)  ::  PDP          ! Dynamic TKE production term
REAL, DIMENSION(D%NIJT,D%NKT), INTENT(IN)   ::  PTP          ! Thermal TKE production term
REAL, DIMENSION(D%NIJT), INTENT(IN),OPTIONAL   ::  PSSUFL_C  ! Time evol Flux of U at sea surface (LOCEAN)
REAL, DIMENSION(D%NIJT), INTENT(IN),OPTIONAL   ::  PSSVFL_C  !
REAL, DIMENSION(D%NIJT), INTENT(IN),OPTIONAL   ::  PSSUFL   
REAL, DIMENSION(D%NIJT), INTENT(IN),OPTIONAL   ::  PSSVFL  !
!
!
!
!
!*       0.2  declaration of local variables
!
!
REAL, DIMENSION(D%NIJT)  :: ZDIRSINZW ! sinus of the angle
                   ! between the normal and the vertical at the surface
REAL, DIMENSION(D%NIJT):: ZCOEFS, &    ! coeff. for the implicit scheme for the wind at the surface
                          ZWORK11D,ZWORK21D,ZWORK31D,ZWORK41D,ZWORK51D,ZWORK61D
REAL, DIMENSION(D%NIJT,D%NKT)  ::  &
       ZA, &       ! under diagonal elements of the tri-diagonal matrix involved
                   ! in the temporal implicit scheme (also used to store coefficient
                   ! J in Section 5)
       ZRES, &     ! guess of the treated variable at t+ deltat when the turbu-
                   ! lence is the only source of evolution added to the ones
                   ! considered in ZSOURCE
       ZFLXZ,  &   ! vertical flux of the treated variable
       ZSOURCE,  & ! source of evolution for the treated variable
       ZKEFF,    & ! effectif diffusion coeff = LT * SQRT( TKE )
       ZWORK1,ZWORK2,&
       ZWORK3,ZWORK4,&
       ZWORK5,ZWORK6! working var. for shuman operators (array syntax)
!
INTEGER             :: IIJE,IIJB,IKB,IKE,IKA,IKU ! index value for the mass points of the domain 
INTEGER             :: IKT          ! array size in k direction
INTEGER             :: IKTB,IKTE    ! start, end of k loops in physical domain
INTEGER             :: JSV,JIJ,JK          ! scalar loop counter
INTEGER             :: IKL
REAL, DIMENSION(D%NIJT)   :: ZCOEFFLXU, &
                             ZCOEFFLXV, ZUSLOPEM, ZVSLOPEM, &
                             ZFLUXSFCU,ZFLUXSFCV
                                    ! coefficients for the surface flux
                                    ! evaluation and copy of PUSLOPEM and
                                    ! PVSLOPEM in local 3D arrays
!
REAL :: ZTIME1, ZTIME2, ZCMFS
TYPE(TFIELDMETADATA) :: TZFIELD
!----------------------------------------------------------------------------
!
!*       1.   PRELIMINARIES
!             -------------
REAL(KIND=JPRB) :: ZHOOK_HANDLE
IF (LHOOK) CALL DR_HOOK('TURB_VER_DYN_FLUX',0,ZHOOK_HANDLE)
!
ZA(:,:)=XUNDEF
PDP(:,:)=XUNDEF
!
IKT=D%NKT
IKTB=D%NKTB
IKTE=D%NKTE
IKB=D%NKB
IKE=D%NKE
IKA=D%NKA
IKU=D%NKU
IKL=D%NKL
IIJE=D%NIJE
IIJB=D%NIJB
!
ZSOURCE(:,:) = 0.
ZFLXZ(:,:) = 0.
ZCMFS = CSTURB%XCMFS
IF (TURBN%LHARAT) ZCMFS=1.
!
!$mnh_expand_array(JIJ=IIJB:IIJE)
ZDIRSINZW(:) = SQRT(1.-PDIRCOSZW(:)**2)
!$mnh_end_expand_array(JIJ=IIJB:IIJE)
!  compute the coefficients for the uncentred gradient computation near the
!  ground
!
! With TURBN%LHARATU length scale and TKE are at half levels so remove MZM
!
IF (TURBN%LHARAT) THEN
  !$mnh_expand_array(JIJ=IIJB:IIJE,JK=1:IKT)
  ZKEFF(:,:) =  PLM(:,:) * SQRT(PTKEM(:,:))
  !$mnh_end_expand_array(JIJ=IIJB:IIJE,JK=1:IKT)
ELSE
  !$mnh_expand_array(JIJ=IIJB:IIJE,JK=1:IKT)
  ZWORK1(:,:) = PLM(:,:) * SQRT(PTKEM(:,:))
  !$mnh_end_expand_array(JIJ=IIJB:IIJE,JK=1:IKT)
  CALL MZM_PHY(D,ZWORK1,ZKEFF)
ENDIF
!
ZUSLOPEM(:)=PUSLOPEM(:)
ZVSLOPEM(:)=PVSLOPEM(:)
ZFLUXSFCU(:)=PSFUM(:)
ZFLUXSFCV(:)=PSFVM(:)
!
!----------------------------------------------------------------------------
!
!
!*       5.   SOURCES OF U,W WIND COMPONENTS AND PARTIAL DYNAMIC PRODUCTION
!             -------------------------------------------------------------
!
!*       5.1  Source of U wind component
!
! Preparation of the arguments for TRIDIAG_WIND
!
CALL MXM_PHY(D,ZKEFF,ZWORK1)
CALL MXM_PHY(D,PDZZ,ZWORK2)
CALL MZM_PHY(D,PRHODJ,ZWORK3)
CALL MXM_PHY(D,ZWORK3,ZWORK4)
!$mnh_expand_array(JIJ=IIJB:IIJE,JK=1:IKT)
ZA(:,:) = -PTSTEP * ZCMFS * ZWORK1(:,:)* ZWORK4(:,:) &
                              / ZWORK2(:,:)**2
!$mnh_end_expand_array(JIJ=IIJB:IIJE,JK=1:IKT)
!
!
! Compute the source of U wind component
!
! compute the coefficient between the vertical flux and the 2 components of the
! wind following the slope
!$mnh_expand_array(JIJ=IIJB:IIJE)
ZCOEFFLXU(:) = PCDUEFF(:) * (PDIRCOSZW(:)**2 - ZDIRSINZW(:)**2) &
                                   * PCOSSLOPE(:)
ZCOEFFLXV(:) = PCDUEFF(:) * PDIRCOSZW(:) * PSINSLOPE(:)
!
! prepare the implicit scheme coefficients for the surface flux
ZCOEFS(:)=  ZCOEFFLXU(:) * PCOSSLOPE(:) * PDIRCOSZW(:)  &
                 +ZCOEFFLXV(:) * PSINSLOPE(:)
!
! average this flux to be located at the U,W vorticity point
!$mnh_end_expand_array(JIJ=IIJB:IIJE)
ZWORK11D(:)=ZCOEFS(:) / PDZZ(:,IKB) 
CALL MXM2D_PHY(D,ZWORK11D,ZCOEFS)
!
!
ZSOURCE(:,IKTB+1:IKTE-1) = 0.
! ZSOURCE= sfc FLUX /DZ
! Sfx flux assumed to be in SI & at vorticity point
CALL MXM_PHY(D,PRHODJ,ZWORK1)
!
IF (OOCEAN) THEN  ! Ocean model
  !$mnh_expand_array(JIJ=IIJB:IIJE)
  ZWORK21D(:) = ZFLUXSFCU(:)/PDZZ(:,IKE)
  !$mnh_end_expand_array(JIJ=IIJB:IIJE)
  CALL MXM2D_PHY(D,ZWORK21D,ZWORK31D)
  !
  !$mnh_expand_array(JIJ=IIJB:IIJE)  
  ZSOURCE(:,IKE) = ZWORK31D(:) &
       *0.5 * ( 1. + ZWORK1(:,IKU) / ZWORK1(:,IKE)) 
  !$mnh_end_expand_array(JIJ=IIJB:IIJE)
  !
  ! Zero flux at the ocean domain bottom
  ZSOURCE(:,IKB) = 0.
  !
ELSE ! Atmosphere
  ! Compute the explicit tangential flux at the W point    
  !$mnh_expand_array(JIJ=IIJB:IIJE)               
  ZSOURCE(:,IKB)     =                                              &
   PTAU11M(:) * PCOSSLOPE(:) * PDIRCOSZW(:) * ZDIRSINZW(:) &
   -PTAU12M(:) * PSINSLOPE(:) * ZDIRSINZW(:)                  &
   -PTAU33M(:) * PCOSSLOPE(:) * ZDIRSINZW(:) * PDIRCOSZW(:)  
!
  ! add the vertical part or the surface flux at the U,W vorticity point
!
  ZWORK31D(:) = ZSOURCE(:,IKB)/PDZZ(:,IKB)
  !$mnh_end_expand_array(JIJ=IIJB:IIJE)
  CALL MXM2D_PHY(D,ZWORK31D,ZWORK41D)
  ZWORK51D(:)= ZCOEFFLXU(:) / PDZZ(:,IKB)       &
         *ZUSLOPEM(:)                           &
        -ZCOEFFLXV(:) / PDZZ(:,IKB)       &
         *ZVSLOPEM(:)
  CALL MXM2D_PHY(D,ZWORK51D,ZWORK61D)
  !$mnh_expand_array(JIJ=IIJB:IIJE)
  ZSOURCE(:,IKB) =                                  &
  (   ZWORK41D(:) &
  +  ZWORK61D(:)   &
  -  ZCOEFS(:) * PUM(:,IKB) * TURBN%XIMPL        &
  ) * 0.5 * ( 1. + ZWORK1(:,IKA) / ZWORK1(:,IKB) )
  !$mnh_end_expand_array(JIJ=IIJB:IIJE)
!
  ZSOURCE(:,IKE) = 0.
ENDIF
!
! Obtention of the split U at t+ deltat
!
CALL TRIDIAG_WIND(D,PUM,ZA,ZCOEFS,PTSTEP,PEXPL,TURBN%XIMPL,   &
                  ZWORK1,ZSOURCE,ZRES)
!
!  Compute the equivalent tendency for the U wind component
!
CALL MXM_PHY(D,PRHODJ,ZWORK1)
CALL MXM_PHY(D,ZKEFF,ZWORK2)
!$mnh_expand_array(JIJ=IIJB:IIJE,JK=1:IKT)
ZWORK3(:,:)=TURBN%XIMPL*ZRES(:,:) + PEXPL*PUM(:,:)
!$mnh_end_expand_array(JIJ=IIJB:IIJE,JK=1:IKT)
CALL DZM_PHY(D,ZWORK3,ZWORK4)
CALL MXM_PHY(D,PDZZ,ZWORK5)
!$mnh_expand_array(JIJ=IIJB:IIJE,JK=1:IKT)
PRUS(:,:)= PRUS(:,:)+ZWORK1(:,:)*(ZRES(:,:) & 
                              - PUM(:,:))/PTSTEP
!
!*       5.2  Partial TKE Dynamic Production
!
! vertical flux of the U wind component
!
ZFLXZ(:,:)     = -ZCMFS * ZWORK2(:,:) * ZWORK4(:,:) &
                                   / ZWORK5(:,:)
!$mnh_end_expand_array(JIJ=IIJB:IIJE,JK=1:IKT)
!
IF (OOCEAN) THEN
  ZFLXZ(:,IKE+1) = ZFLUXSFCU(:)
ELSE
  ! surface flux
  CALL MXM_PHY(D,PDZZ,ZWORK1)
  CALL MXM_PHY(D,PRHODJ,ZWORK2)
  !$mnh_expand_array(JIJ=IIJB:IIJE)
  ZFLXZ(:,IKB)   =   ZWORK1(:,IKB)  *                &
    ( ZSOURCE(:,IKB)                                         &
     +ZCOEFS(:) * ZRES(:,IKB) * TURBN%XIMPL                &
    ) / 0.5 / ( 1. + ZWORK2(:,IKA)/ ZWORK2(:,IKB) )
  !
  ZFLXZ(:,IKA) = ZFLXZ(:,IKB)
  !$mnh_end_expand_array(JIJ=IIJB:IIJE)
END IF
!
CALL MY_UNPACK(ZFLXZ,TDUMMY10)
IF ( TURBN%LTURB_FLX .AND. TPFILE%LOPENED ) THEN
  ! stores the U wind component vertical flux
  TZFIELD = TFIELDMETADATA(                        &
    CMNHNAME   = 'UW_VFLX',                        &
    CSTDNAME   = '',                               &
    CLONGNAME  = 'UW_VFLX',                        &
    CUNITS     = 'm2 s-2',                         &
    CDIR       = 'XY',                             &
    CCOMMENT   = 'U wind component vertical flux', &
    NGRID      = 4,                                &
    NTYPE      = TYPEREAL,                         &
    NDIMS      = 3,                                &
    LTIMEDEP   = .TRUE.                            )
  CALL IO_FIELD_WRITE_PHY(D,TPFILE,TZFIELD,ZFLXZ)
END IF
!
! first part of total momentum flux
!
PWU(:,:) = ZFLXZ(:,:)
!
! Contribution to the TKE dynamic production of TKE
! (computed at mass point)
!
CALL GZ_U_UW_PHY(D,PUM,PDZZ,ZWORK1)
!$mnh_expand_array(JIJ=IIJB:IIJE,JK=1:IKT)
ZWORK2(:,:) = ZFLXZ(:,:) * ZWORK1(:,:)
!$mnh_end_expand_array(JIJ=IIJB:IIJE,JK=1:IKT)
CALL MXF_PHY(D,ZWORK2,ZWORK3)
CALL MZF_PHY(D,ZWORK3,ZWORK4)
!$mnh_expand_array(JIJ=IIJB:IIJE,JK=1:IKT)
PDP(:,:) = -ZWORK4(:,:)
!$mnh_end_expand_array(JIJ=IIJB:IIJE,JK=1:IKT)
!
! Special cases near surface
CALL MXM_PHY(D,PDZZ,ZWORK1)
IF (OOCEAN) THEN
  ! evaluate the dynamic production at w(IKE) and store in PDP(IKE)
  ! before to be extrapolated in tke_eps routine
  !$mnh_expand_array(JIJ=IIJB:IIJE)  
  ZWORK2(:,IKE) = ZFLXZ(:,IKE) * (PUM(:,IKE)-PUM(:,IKE-IKL))  &
                         / ZWORK1(:,IKE-IKL)
  !$mnh_end_expand_array(JIJ=IIJB:IIJE)                 
  CALL MXF_PHY(D,ZWORK2,ZWORK3)
  !$mnh_expand_array(JIJ=IIJB:IIJE)  
  PDP(:,IKE) = -ZWORK3(:,IKE)
  !$mnh_end_expand_array(JIJ=IIJB:IIJE)
ELSE ! Atmosphere
  ! evaluate the dynamic production at w(IKB+KKL) in PDP(IKB)
  !$mnh_expand_array(JIJ=IIJB:IIJE)  
  ZWORK2(:,IKB) = ZFLXZ(:,IKB+IKL) * (PUM(:,IKB+IKL)-PUM(:,IKB))  &
                           / ZWORK1(:,IKB+IKL)
  !$mnh_end_expand_array(JIJ=IIJB:IIJE)                   
  CALL MXF_PHY(D,ZWORK2,ZWORK3)
  !$mnh_expand_array(JIJ=IIJB:IIJE)
  PDP(:,IKB) = -ZWORK3(:,IKB)
  !$mnh_end_expand_array(JIJ=IIJB:IIJE)
!
END IF
!
! Storage in the LES configuration
!
IF (TLES%LLES_CALL) THEN
  CALL SECOND_MNH(ZTIME1)
  !
  CALL MXF_PHY(D,ZFLXZ,ZWORK1)
  CALL MZF_PHY(D,ZWORK1,ZWORK2)
  CALL LES_MEAN_SUBGRID_PHY(D,TLES,ZWORK2, TLES%X_LES_SUBGRID_WU )
  !
  CALL GZ_U_UW_PHY(D,PUM,PDZZ,ZWORK1)
  !$mnh_expand_array(JIJ=IIJB:IIJE,JK=1:IKT)
  ZWORK1(:,:) = ZWORK1(:,:) * ZFLXZ(:,:)
  !$mnh_end_expand_array(JIJ=IIJB:IIJE,JK=1:IKT)
  CALL MXF_PHY(D,ZWORK1,ZWORK2)
  CALL MZF_PHY(D,ZWORK2,ZWORK3) 
  CALL LES_MEAN_SUBGRID_PHY(D,TLES,ZWORK3, TLES%X_LES_RES_ddxa_U_SBG_UaU )
  !
  !$mnh_expand_array(JIJ=IIJB:IIJE,JK=1:IKT)
  ZWORK1(:,:) = ZCMFS * ZKEFF(:,:)
  !$mnh_end_expand_array(JIJ=IIJB:IIJE,JK=1:IKT)
  CALL LES_MEAN_SUBGRID_PHY(D,TLES, ZWORK1, TLES%X_LES_SUBGRID_Km )
  !
  CALL SECOND_MNH(ZTIME2)
  TLES%XTIME_LES = TLES%XTIME_LES + ZTIME2 - ZTIME1
END IF
!
!*       5.3  Source of W wind component
!
!
IF(TURBN%CTURBDIM=='3DIM') THEN
  ! Compute the source for the W wind component
                ! used to compute the W source at the ground
  !$mnh_expand_array(JIJ=IIJB:IIJE)
  ZFLXZ(:,IKA) = 2 * ZFLXZ(:,IKB) - ZFLXZ(:,IKB+IKL) ! extrapolation 
  !$mnh_end_expand_array(JIJ=IIJB:IIJE)
 IF (OOCEAN) THEN
   !$mnh_expand_array(JIJ=IIJB:IIJE)
   ZFLXZ(:,IKU) = 2 * ZFLXZ(:,IKE) - ZFLXZ(:,IKE-IKL) ! extrapolation
   !$mnh_end_expand_array(JIJ=IIJB:IIJE)
 END IF
  !
  CALL MXM_PHY(D,PRHODJ,ZWORK1)
  !$mnh_expand_array(JIJ=IIJB:IIJE,JK=1:IKT)
  ZWORK1(:,:) = ZWORK1(:,:) / PDXX(:,:)
  !$mnh_end_expand_array(JIJ=IIJB:IIJE,JK=1:IKT)
  CALL MZM_PHY(D,ZWORK1,ZWORK2)
  !$mnh_expand_array(JIJ=IIJB:IIJE,JK=1:IKT)
  ZWORK2(:,:) = ZWORK2(:,:) * ZFLXZ(:,:)
  !$mnh_end_expand_array(JIJ=IIJB:IIJE,JK=1:IKT)
  CALL DXF_PHY(D,ZWORK2,ZWORK1)
  !
  IF (.NOT. OFLAT) THEN
    !
    !$mnh_expand_array(JIJ=IIJB:IIJE,JK=1:IKT)
    ZWORK2(:,:) = ZFLXZ(:,:)*PDZX(:,:)
    !$mnh_end_expand_array(JIJ=IIJB:IIJE,JK=1:IKT)
    CALL MZF_PHY(D,ZWORK2,ZWORK3)
    !$mnh_expand_array(JIJ=IIJB:IIJE,JK=1:IKT)
    ZWORK3(:,:) = ZWORK3(:,:) / PDXX(:,:)
    !$mnh_end_expand_array(JIJ=IIJB:IIJE,JK=1:IKT)
    CALL MXF_PHY(D,ZWORK3,ZWORK2)
    CALL MZF_PHY(D,PDZZ,ZWORK3)
    !$mnh_expand_array(JIJ=IIJB:IIJE,JK=1:IKT)
    ZWORK3(:,:) = PRHODJ(:,:) &
                                             / ZWORK3(:,:) * ZWORK2(:,:)
    !$mnh_end_expand_array(JIJ=IIJB:IIJE,JK=1:IKT)
    CALL DZM_PHY(D,ZWORK3,ZWORK2)
    !$mnh_expand_array(JIJ=IIJB:IIJE,JK=1:IKT)
    PRWS(:,:) = PRWS(:,:) - ZWORK1(:,:) &
                                  + ZWORK2(:,:)
    !$mnh_end_expand_array(JIJ=IIJB:IIJE,JK=1:IKT)
  ELSE
    !$mnh_expand_array(JIJ=IIJB:IIJE,JK=1:IKT)
    PRWS(:,:)= PRWS(:,:) - ZWORK1(:,:)
    !$mnh_end_expand_array(JIJ=IIJB:IIJE,JK=1:IKT)
  END IF
  !
  ! Complete the TKE dynamical production with the W wind contribution 
  !
  CALL GX_W_UW_PHY(D,OFLAT,PWM,PDXX,PDZZ,PDZX, ZWORK1)
  !$mnh_expand_array(JIJ=IIJB:IIJE,JK=1:IKT)
  ZWORK1(:,:) = ZWORK1(:,:) * ZFLXZ(:,:)
  !$mnh_end_expand_array(JIJ=IIJB:IIJE,JK=1:IKT)
  CALL MXF_PHY(D,ZWORK1,ZWORK2)
  CALL MZF_PHY(D,ZWORK2,ZWORK3)
  !$mnh_expand_array(JIJ=IIJB:IIJE,JK=1:IKT)
  ZA(:,:) = -ZWORK3(:,:)
  !$mnh_end_expand_array(JIJ=IIJB:IIJE,JK=1:IKT)
  !
  ! Special cases near surface
  CALL DXM_PHY(D,PWM,ZWORK1)
  IF (OOCEAN) THEN
    ! evaluate the dynamic production at w(IKE) in PDP(IKE)
    !
    !$mnh_expand_array(JIJ=IIJB:IIJE)
    ZWORK31D(:) = - ZFLXZ(:,IKE) *  ZWORK1(:,IKE) &
                            / (0.5*(PDXX(:,IKE-IKL)+PDXX(:,IKE)))
    !$mnh_end_expand_array(JIJ=IIJB:IIJE)
    CALL MXF2D_PHY(D,ZWORK31D,ZWORK41D)
    ZA(:,IKE) = ZWORK41D(:)
  !
  ELSE !Atmosphere
    ! evaluate the dynamic production at w(IKB+IKL) in PDP(IKB)
    !
    !$mnh_expand_array(JIJ=IIJB:IIJE)
    ZWORK21D(:) = (PWM(:,IKB+2*IKL)-PWM(:,IKB+IKL)) &
                          / (PDZZ(:,IKB+2*IKL)+PDZZ(:,IKB+IKL))       &
                          + (PWM(:,IKB+IKL)-PWM(:,IKB))                       &
                          / (PDZZ(:,IKB+IKL)+PDZZ(:,IKB)) 
    !$mnh_end_expand_array(JIJ=IIJB:IIJE)
    !
    CALL MXM2D_PHY(D,ZWORK21D,ZWORK51D)
    !$mnh_expand_array(JIJ=IIJB:IIJE)
    ZWORK31D(:) = - ZFLXZ(:,IKB+IKL) &
                                      * ( ZWORK1(:,IKB+IKL) - ZWORK51D(:) &
                                      *   PDZX(:,IKB+IKL) ) &
                                      / (0.5*(PDXX(:,IKB+IKL)+PDXX(:,IKB)))
    !$mnh_end_expand_array(JIJ=IIJB:IIJE)
    CALL MXF2D_PHY(D,ZWORK31D,ZWORK41D)
    ZA(:,IKB) = ZWORK41D(:)
    !
  END IF
  !
  !$mnh_expand_array(JIJ=IIJB:IIJE,JK=1:IKT)
  PDP(:,:)=PDP(:,:)+ZA(:,:)
  !$mnh_end_expand_array(JIJ=IIJB:IIJE,JK=1:IKT)
  !
  ! Storage in the LES configuration
  !
  IF (TLES%LLES_CALL) THEN
    CALL SECOND_MNH(ZTIME1)
    !
    CALL GX_W_UW_PHY(D,OFLAT,PWM,PDXX,PDZZ,PDZX,ZWORK1)
    !$mnh_expand_array(JIJ=IIJB:IIJE,JK=1:IKT)
    ZWORK1(:,:) = ZWORK1(:,:)*ZFLXZ(:,:)
    !$mnh_end_expand_array(JIJ=IIJB:IIJE,JK=1:IKT)
    CALL MXF_PHY(D,ZWORK1,ZWORK2)
    CALL MZF_PHY(D,ZWORK2,ZWORK1)
    CALL LES_MEAN_SUBGRID_PHY(D,TLES,ZWORK1, TLES%X_LES_RES_ddxa_W_SBG_UaW )
    !
    CALL GX_M_U_PHY(D,OFLAT,PTHLM,PDXX,PDZZ,PDZX,ZWORK1)
    CALL MZF_PHY(D,ZFLXZ,ZWORK2)
    !$mnh_expand_array(JIJ=IIJB:IIJE,JK=1:IKT)
    ZWORK2(:,:) = ZWORK2(:,:) * ZWORK1(:,:)
    !$mnh_end_expand_array(JIJ=IIJB:IIJE,JK=1:IKT)
    CALL MXF_PHY(D,ZWORK2,ZWORK1)
    CALL LES_MEAN_SUBGRID_PHY(D,TLES,ZWORK1, TLES%X_LES_RES_ddxa_Thl_SBG_UaW )
    !
    IF (KRR>=1) THEN
      CALL GX_U_M_PHY(D,OFLAT,PRM(:,:,1),PDXX,PDZZ,PDZX,ZWORK1)
      CALL MZF_PHY(D,ZFLXZ,ZWORK2)
      !$mnh_expand_array(JIJ=IIJB:IIJE,JK=1:IKT)
      ZWORK1(:,:) = ZWORK1(:,:) * ZWORK2(:,:)
      !$mnh_end_expand_array(JIJ=IIJB:IIJE,JK=1:IKT)
      CALL MXF_PHY(D,ZWORK1,ZWORK2)
      CALL LES_MEAN_SUBGRID_PHY(D,TLES,ZWORK2,TLES%X_LES_RES_ddxa_Rt_SBG_UaW )
    END IF
    DO JSV=1,KSV
      CALL GX_U_M_PHY(D,OFLAT,PSVM(:,:,JSV),PDXX,PDZZ,PDZX,ZWORK1)
      CALL MZF_PHY(D,ZFLXZ,ZWORK2)
      !$mnh_expand_array(JIJ=IIJB:IIJE,JK=1:IKT)
      ZWORK1(:,:) = ZWORK1(:,:) * ZWORK2(:,:)
      !$mnh_end_expand_array(JIJ=IIJB:IIJE,JK=1:IKT)
      CALL MXF_PHY(D,ZWORK1,ZWORK2)
      CALL LES_MEAN_SUBGRID_PHY(D,TLES,ZWORK2,TLES%X_LES_RES_ddxa_Sv_SBG_UaW(:,:,:,JSV) )
    END DO
    CALL SECOND_MNH(ZTIME2)
    TLES%XTIME_LES = TLES%XTIME_LES + ZTIME2 - ZTIME1
  END IF
END IF
!
!----------------------------------------------------------------------------
!
!
!*       6.   SOURCES OF V,W WIND COMPONENTS AND COMPLETE 1D TKE DYNAMIC PRODUCTION 
!             -----------------------------------------------------------------
!
!*       6.1  Source of V wind component
!
! Preparation of the arguments for TRIDIAG_WIND
!!
CALL MYM_PHY(D,ZKEFF,ZWORK1)
CALL MYM_PHY(D,PDZZ,ZWORK2)
CALL MZM_PHY(D,PRHODJ,ZWORK3)
CALL MYM_PHY(D,ZWORK3,ZWORK4)
!$mnh_expand_array(JIJ=IIJB:IIJE,JK=1:IKT)
ZA(:,:) = -PTSTEP * ZCMFS * ZWORK1(:,:)* ZWORK4(:,:) & 
                              / ZWORK2(:,:)**2
!$mnh_end_expand_array(JIJ=IIJB:IIJE,JK=1:IKT)
!
!
!
! Compute the source of V wind component
! compute the coefficient between the vertical flux and the 2 components of the
! wind following the slope
!$mnh_expand_array(JIJ=IIJB:IIJE)
ZCOEFFLXU(:) = PCDUEFF(:) * (PDIRCOSZW(:)**2 - ZDIRSINZW(:)**2) &
                                   * PSINSLOPE(:)
ZCOEFFLXV(:) = PCDUEFF(:) * PDIRCOSZW(:) * PCOSSLOPE(:)

! prepare the implicit scheme coefficients for the surface flux
ZCOEFS(:)=  ZCOEFFLXU(:) * PSINSLOPE(:) * PDIRCOSZW(:)  &
               +ZCOEFFLXV(:) * PCOSSLOPE(:)
!$mnh_end_expand_array(JIJ=IIJB:IIJE)
!
! average this flux to be located at the V,W vorticity point
!$mnh_expand_array(JIJ=IIJB:IIJE)
ZWORK11D(:)=ZCOEFS(:) / PDZZ(:,IKB) 
!$mnh_end_expand_array(JIJ=IIJB:IIJE)
CALL MYM2D_PHY(D,ZWORK11D,ZCOEFS)
!
! No flux in SOURCE TERM NULL OUTSIDE BC 
ZSOURCE(:,IKB+1:IKE-1) = 0.
! Surface case
CALL MYM_PHY(D,PRHODJ,ZWORK1)
IF (OOCEAN) THEN ! Ocean case
  ZCOEFFLXU(:) = PCDUEFF(:)
  ZCOEFFLXV(:) = PCDUEFF(:)
  ZCOEFS(:)=ZCOEFFLXU(:)
  ! average this flux to be located at the U,W vorticity point
  !$mnh_expand_array(JIJ=IIJB:IIJE)  
  ZWORK11D(:) = ZCOEFS(:) / PDZZ(:,IKE)
  !$mnh_end_expand_array(JIJ=IIJB:IIJE)  
  CALL MYM2D_PHY(D,ZWORK11D,ZCOEFS)
  !
  ZWORK11D(:) = ZFLUXSFCV(:) / PDZZ(:,IKE)
  CALL MYM2D_PHY(D,ZWORK11D,ZWORK21D)
  !
  !$mnh_expand_array(JIJ=IIJB:IIJE)
    ZSOURCE(:,IKE) = ZWORK21D(:) &
        *0.5 * ( 1. + ZWORK1(:,IKU) / ZWORK1(:,IKE))
  !$mnh_end_expand_array(JIJ=IIJB:IIJE)
  !No flux at the ocean domain bottom
  ZSOURCE(:,IKB) = 0.
!
ELSE ! Atmos case
!
  !$mnh_expand_array(JIJ=IIJB:IIJE)
  ZWORK31D(:) = ZCOEFFLXU(:) / PDZZ(:,IKB) &
            *ZUSLOPEM(:)                                   &
            +ZCOEFFLXV(:) / PDZZ(:,IKB)            &
            *ZVSLOPEM(:) 
  !$mnh_end_expand_array(JIJ=IIJB:IIJE)
  CALL MYM2D_PHY(D,ZWORK31D,ZWORK61D)
  !
  ! compute the explicit tangential flux at the W point
  !$mnh_expand_array(JIJ=IIJB:IIJE)
  ZSOURCE(:,IKB) =                                                                    &
    PTAU11M(:) * PSINSLOPE(:) * PDIRCOSZW(:) * ZDIRSINZW(:)   &
   +PTAU12M(:) * PCOSSLOPE(:) * ZDIRSINZW(:)                          &
   -PTAU33M(:) * PSINSLOPE(:) * ZDIRSINZW(:) * PDIRCOSZW(:) 
  !
  ZWORK31D(:) = ZSOURCE(:,IKB)/PDZZ(:,IKB)
  !$mnh_end_expand_array(JIJ=IIJB:IIJE)
  CALL MYM2D_PHY(D,ZWORK31D,ZWORK51D)
!
  ! add the vertical part or the surface flux at the V,W vorticity point
  !$mnh_expand_array(JIJ=IIJB:IIJE)
  ZSOURCE(:,IKB) =                                      &
  (  ZWORK51D(:)                                        &
   + ZWORK61D(:)                                        &
   - ZCOEFS(:) * PVM(:,IKB) * TURBN%XIMPL             &
  ) * 0.5 * ( 1. + ZWORK1(:,IKA) / ZWORK1(:,IKB) )
  !$mnh_end_expand_array(JIJ=IIJB:IIJE)
!
  !No flux at the atmosphere top
  ZSOURCE(:,IKE) = 0.
ENDIF ! End of Ocean or Atmospher Cases
! 
!  Obtention of the split V at t+ deltat 
CALL TRIDIAG_WIND(D,PVM,ZA,ZCOEFS,PTSTEP,PEXPL,TURBN%XIMPL,  &
                  ZWORK1,ZSOURCE,ZRES)
!
! Compute the equivalent tendency for the V wind component
!
CALL MYM_PHY(D,PRHODJ,ZWORK1)
CALL MYM_PHY(D,ZKEFF,ZWORK2)
!$mnh_expand_array(JIJ=IIJB:IIJE,JK=1:IKT)
ZWORK3(:,:)=TURBN%XIMPL*ZRES(:,:) + PEXPL*PVM(:,:)
!$mnh_end_expand_array(JIJ=IIJB:IIJE,JK=1:IKT)
CALL DZM_PHY(D,ZWORK3,ZWORK4)
CALL MYM_PHY(D,PDZZ,ZWORK5)
!$mnh_expand_array(JIJ=IIJB:IIJE,JK=1:IKT)
PRVS(:,:) = PRVS(:,:)+ZWORK1(:,:)*(ZRES(:,:)& 
                               - PVM(:,:))/PTSTEP
!
!
!*       6.2  Complete 1D dynamic Production
!
!  vertical flux of the V wind component
!
ZFLXZ(:,:)   = -ZCMFS * ZWORK2(:,:) * ZWORK4(:,:) &
                                   / ZWORK5(:,:)
!$mnh_end_expand_array(JIJ=IIJB:IIJE,JK=1:IKT)
!
IF (OOCEAN) THEN
  ZFLXZ(:,IKE+1)  = ZFLUXSFCV(:)
ELSE
  !$mnh_expand_array(JIJ=IIJB:IIJE)
  ZFLXZ(:,IKB)   =   ZWORK5(:,IKB)  *                &
    ( ZSOURCE(:,IKB)                                         &
     +ZCOEFS(:) * ZRES(:,IKB) * TURBN%XIMPL                &
    ) / 0.5 / ( 1. + ZWORK1(:,IKA) / ZWORK1(:,IKB) )
  !
  ZFLXZ(:,IKA) = ZFLXZ(:,IKB)
  !$mnh_end_expand_array(JIJ=IIJB:IIJE)
END IF
!
CALL MY_UNPACK(ZFLXZ,TDUMMY11)
IF ( TURBN%LTURB_FLX .AND. TPFILE%LOPENED ) THEN
  ! stores the V wind component vertical flux
  TZFIELD = TFIELDMETADATA(                        &
    CMNHNAME   = 'VW_VFLX',                        &
    CSTDNAME   = '',                               &
    CLONGNAME  = 'VW_VFLX',                        &
    CUNITS     = 'm2 s-2',                         &
    CDIR       = 'XY',                             &
    CCOMMENT   = 'V wind component vertical flux', &
    NGRID      = 4,                                &
    NTYPE      = TYPEREAL,                         &
    NDIMS      = 3,                                &
    LTIMEDEP   = .TRUE.                            )
  CALL IO_FIELD_WRITE_PHY(D,TPFILE,TZFIELD,ZFLXZ)
END IF
!
! second part of total momentum flux
!
PWV(:,:) = ZFLXZ(:,:)
!
!  Contribution to the TKE dynamical production 
!    computed at mass point
!
CALL GZ_V_VW_PHY(D,PVM,PDZZ,ZWORK1)
!$mnh_expand_array(JIJ=IIJB:IIJE,JK=1:IKT)
ZWORK2(:,:) = ZFLXZ(:,:) * ZWORK1(:,:)
!$mnh_end_expand_array(JIJ=IIJB:IIJE,JK=1:IKT)
CALL MYF_PHY(D,ZWORK2,ZWORK3)
CALL MZF_PHY(D,ZWORK3,ZWORK4)
!$mnh_expand_array(JIJ=IIJB:IIJE,JK=1:IKT)
ZA(:,:) = -ZWORK4(:,:)
!$mnh_end_expand_array(JIJ=IIJB:IIJE,JK=1:IKT)
!
! Special cases at surface
CALL MYM_PHY(D,PDZZ,ZWORK1)
IF (OOCEAN) THEN
  ! evaluate the dynamic production at w(IKE) in PDP(IKE)
  ! before extrapolation done in routine tke_eps_source
  !$mnh_expand_array(JIJ=IIJB:IIJE)
  ZWORK2(:,IKE) = ZFLXZ(:,IKE) * (PVM(:,IKE)-PVM(:,IKE-IKL))  &
                         / ZWORK1(:,IKE-IKL)
  !$mnh_end_expand_array(JIJ=IIJB:IIJE)
  CALL MYF_PHY(D,ZWORK2,ZWORK3)
  !$mnh_expand_array(JIJ=IIJB:IIJE)
  ZA(:,IKE) = -ZWORK3(:,IKE)
  !$mnh_end_expand_array(JIJ=IIJB:IIJE)
!
ELSE ! Atmosphere
  ! evaluate the dynamic production at w(IKB+IKL) in PDP(IKB)
  !$mnh_expand_array(JIJ=IIJB:IIJE)
  ZWORK2(:,IKB) = ZFLXZ(:,IKB+IKL) * (PVM(:,IKB+IKL)-PVM(:,IKB))  &
                           / ZWORK1(:,IKB+IKL)
  !$mnh_end_expand_array(JIJ=IIJB:IIJE)
  CALL MYF_PHY(D,ZWORK2,ZWORK3)
  !$mnh_expand_array(JIJ=IIJB:IIJE)
  ZA(:,IKB) = -ZWORK3(:,IKB)
  !$mnh_end_expand_array(JIJ=IIJB:IIJE)
END IF
!
!$mnh_expand_array(JIJ=IIJB:IIJE,JK=1:IKT)
PDP(:,:)=PDP(:,:)+ZA(:,:)
!$mnh_end_expand_array(JIJ=IIJB:IIJE,JK=1:IKT)
!
! Storage in the LES configuration
!
IF (TLES%LLES_CALL) THEN
  CALL SECOND_MNH(ZTIME1)
  !
  CALL MYF_PHY(D,ZFLXZ,ZWORK1)
  CALL MZF_PHY(D,ZWORK1,ZWORK2)
  CALL LES_MEAN_SUBGRID_PHY(D,TLES,ZWORK2, TLES%X_LES_SUBGRID_WV )
  !
  CALL GZ_V_VW_PHY(D,PVM,PDZZ,ZWORK1)
  !$mnh_expand_array(JIJ=IIJB:IIJE,JK=1:IKT)
  ZWORK1(:,:) = ZWORK1(:,:) * ZFLXZ(:,:)
  !$mnh_end_expand_array(JIJ=IIJB:IIJE,JK=1:IKT)
  CALL MYF_PHY(D,ZWORK1,ZWORK2)
  CALL MZF_PHY(D,ZWORK2,ZWORK1)
  CALL LES_MEAN_SUBGRID_PHY(D,TLES,ZWORK1, TLES%X_LES_RES_ddxa_V_SBG_UaV )
  !
  CALL SECOND_MNH(ZTIME2)
  TLES%XTIME_LES = TLES%XTIME_LES + ZTIME2 - ZTIME1
END IF
!
!
!*       6.3  Source of W wind component
!
IF(TURBN%CTURBDIM=='3DIM') THEN
  ! Compute the source for the W wind component
  IF (OOCEAN) THEN
    !$mnh_expand_array(JIJ=IIJB:IIJE)
    ZFLXZ(:,IKE+IKL) = 2 * ZFLXZ(:,IKE) - ZFLXZ(:,IKE-IKL) ! extrapolation 
    !$mnh_end_expand_array(JIJ=IIJB:IIJE)
  ELSE
    !$mnh_expand_array(JIJ=IIJB:IIJE)
    ZFLXZ(:,IKA) = 2 * ZFLXZ(:,IKB) - ZFLXZ(:,IKB+IKL) ! extrapolation
    !$mnh_end_expand_array(JIJ=IIJB:IIJE)
  END IF
  !
  IF (.NOT. O2D) THEN
    CALL MYM_PHY(D,PRHODJ,ZWORK1)
    !$mnh_expand_array(JIJ=IIJB:IIJE,JK=1:IKT)
    ZWORK1(:,:) = ZWORK1(:,:) / PDYY(:,:)
    !$mnh_end_expand_array(JIJ=IIJB:IIJE,JK=1:IKT)
    CALL MZM_PHY(D,ZWORK1,ZWORK2)
    !$mnh_expand_array(JIJ=IIJB:IIJE,JK=1:IKT)
    ZWORK2(:,:) = ZWORK2(:,:) * ZFLXZ(:,:)
    !$mnh_end_expand_array(JIJ=IIJB:IIJE,JK=1:IKT)
    CALL DYF_PHY(D,ZWORK2,ZWORK1)
    !
    !ZWORK1 = DYF( MZM(MYM(PRHODJ) /PDYY, IKA, IKU, IKL) * ZFLXZ ) 
    IF (.NOT. OFLAT) THEN
      CALL MZF_PHY(D,PDZZ,ZWORK3)
      !$mnh_expand_array(JIJ=IIJB:IIJE,JK=1:IKT)
      ZWORK2(:,:) = ZFLXZ(:,:) * PDZY(:,:)
      !$mnh_end_expand_array(JIJ=IIJB:IIJE,JK=1:IKT)      
      CALL MZF_PHY(D,ZWORK2,ZWORK4)
      !$mnh_expand_array(JIJ=IIJB:IIJE,JK=1:IKT)      
      ZWORK4(:,:) = ZWORK4(:,:) / PDYY(:,:)
      !$mnh_end_expand_array(JIJ=IIJB:IIJE,JK=1:IKT)      
      CALL MYF_PHY(D,ZWORK4,ZWORK2)
      !$mnh_expand_array(JIJ=IIJB:IIJE,JK=1:IKT)      
      ZWORK3(:,:) = PRHODJ(:,:) / ZWORK3(:,:) &
                                      * ZWORK2(:,:)
      !$mnh_end_expand_array(JIJ=IIJB:IIJE,JK=1:IKT)      
      CALL DZM_PHY(D,ZWORK3,ZWORK2)
      !ZWORK2 = DZM(PRHODJ / MZF(PDZZ) * MYF(MZF(ZFLXZ*PDZY) / PDYY ) )
      !
      !$mnh_expand_array(JIJ=IIJB:IIJE,JK=1:IKT)
      PRWS(:,:) = PRWS(:,:) - ZWORK1(:,:) &
                                    + ZWORK2(:,:)
      !$mnh_end_expand_array(JIJ=IIJB:IIJE,JK=1:IKT)
    ELSE
      !$mnh_expand_array(JIJ=IIJB:IIJE,JK=1:IKT)
      PRWS(:,:)= PRWS(:,:) - ZWORK1(:,:)
      !$mnh_end_expand_array(JIJ=IIJB:IIJE,JK=1:IKT)
    END IF
  END IF
  ! 
  ! Complete the Dynamical production with the W wind component 
  IF (.NOT. O2D) THEN
    CALL GY_W_VW_PHY(D,OFLAT,PWM,PDYY,PDZZ,PDZY, ZWORK1)
    !$mnh_expand_array(JIJ=IIJB:IIJE,JK=1:IKT)
    ZWORK1(:,:) = ZWORK1(:,:) * ZFLXZ(:,:)
    !$mnh_end_expand_array(JIJ=IIJB:IIJE,JK=1:IKT)
    CALL MYF_PHY(D,ZWORK1,ZWORK2)
    CALL MZF_PHY(D,ZWORK2,ZWORK3)
    !$mnh_expand_array(JIJ=IIJB:IIJE,JK=1:IKT)
    ZA(:,:) = -ZWORK3(:,:)
    !$mnh_end_expand_array(JIJ=IIJB:IIJE,JK=1:IKT)
    !
    CALL DYM_PHY(D,PWM,ZWORK1)
    ! Special case near surface 
    IF (OOCEAN) THEN
      ! evaluate the dynamic production at w(IKE) and stored in PDP(IKE)
      !$mnh_expand_array(JIJ=IIJB:IIJE)
      ZWORK31D(:) = - ZFLXZ(:,IKE) *  ZWORK1(:,IKE) &
                            / (0.5*(PDYY(:,IKE-IKL)+PDYY(:,IKE)))
      !$mnh_end_expand_array(JIJ=IIJB:IIJE)
      CALL MYF2D_PHY(D,ZWORK31D,ZWORK41D)
      ZA(:,IKE) = ZWORK41D(:)
    ELSE ! Atmosphere
      ! evaluate the dynamic production at w(IKB+KKL) and stored in PDP(IKB)
      !$mnh_expand_array(JIJ=IIJB:IIJE)
      ZWORK21D(:) = (PWM(:,IKB+2*IKL   )-PWM(:,IKB+IKL)) &
                            / (PDZZ(:,IKB+2*IKL)+PDZZ(:,IKB+IKL))       &
                            + (PWM(:,IKB+IKL)-PWM(:,IKB))                       &
                            / (PDZZ(:,IKB+IKL)+PDZZ(:,IKB)) 
      !$mnh_end_expand_array(JIJ=IIJB:IIJE)
      !
      CALL MYM2D_PHY(D,ZWORK21D,ZWORK51D)
      !$mnh_expand_array(JIJ=IIJB:IIJE)
      ZWORK31D(:  ) = - ZFLXZ(:,IKB+IKL) &
                                        * ( ZWORK1(:,IKB+IKL) - ZWORK51D(:  ) &
                                        *   PDZY(:,IKB+IKL) ) &
                                        / (0.5*(PDYY(:,IKB+IKL)+PDYY(:,IKB)))
      !$mnh_end_expand_array(JIJ=IIJB:IIJE)
      CALL MYF2D_PHY(D,ZWORK31D,ZWORK41D)
      ZA(:,IKB) = ZWORK41D(:)
    !
    END IF
!
    !$mnh_expand_array(JIJ=IIJB:IIJE,JK=1:IKT)
    PDP(:,:)=PDP(:,:)+ZA(:,:)
    !$mnh_end_expand_array(JIJ=IIJB:IIJE,JK=1:IKT)
  !
  END IF
  !
  ! Storage in the LES configuration
  !
  IF (TLES%LLES_CALL) THEN
    CALL SECOND_MNH(ZTIME1)
    !
    CALL GY_W_VW_PHY(D,OFLAT,PWM,PDYY,PDZZ,PDZY,ZWORK1)
    !$mnh_expand_array(JIJ=IIJB:IIJE,JK=1:IKT)
    ZWORK1(:,:) = ZWORK1(:,:)*ZFLXZ(:,:)
    !$mnh_end_expand_array(JIJ=IIJB:IIJE,JK=1:IKT)
    CALL MYF_PHY(D,ZWORK1,ZWORK2)
    CALL MZF_PHY(D,ZWORK2,ZWORK1)
    CALL LES_MEAN_SUBGRID_PHY(D,TLES,ZWORK1,TLES%X_LES_RES_ddxa_W_SBG_UaW , .TRUE. )
    !
    CALL GY_M_V_PHY(D,OFLAT,PTHLM,PDYY,PDZZ,PDZY,ZWORK1)
    CALL MZF_PHY(D,ZFLXZ,ZWORK2)
    !$mnh_expand_array(JIJ=IIJB:IIJE,JK=1:IKT)
    ZWORK2(:,:) = ZWORK2(:,:) * ZWORK1(:,:)
    !$mnh_end_expand_array(JIJ=IIJB:IIJE,JK=1:IKT)
    CALL MYF_PHY(D,ZWORK2,ZWORK1)
    CALL LES_MEAN_SUBGRID_PHY(D,TLES,ZWORK1,TLES%X_LES_RES_ddxa_Thl_SBG_UaW , .TRUE. )
    !
    IF (KRR>=1) THEN
      CALL GY_V_M_PHY(D,OFLAT,PRM(:,:,1),PDYY,PDZZ,PDZY,ZWORK1)
      CALL MZF_PHY(D,ZFLXZ,ZWORK2)
      !$mnh_expand_array(JIJ=IIJB:IIJE,JK=1:IKT)
      ZWORK1(:,:) = ZWORK1(:,:) * ZWORK2(:,:)
      !$mnh_end_expand_array(JIJ=IIJB:IIJE,JK=1:IKT)
      CALL MYF_PHY(D,ZWORK1,ZWORK2)
      CALL LES_MEAN_SUBGRID_PHY(D,TLES,ZWORK2,TLES%X_LES_RES_ddxa_Rt_SBG_UaW , .TRUE. )
    END IF
    !
    CALL SECOND_MNH(ZTIME2)
    TLES%XTIME_LES = TLES%XTIME_LES + ZTIME2 - ZTIME1
  END IF
  !
END IF
!
!
!----------------------------------------------------------------------------
!
!*       7.   DIAGNOSTIC COMPUTATION OF THE 1D <W W> VARIANCE
!             -----------------------------------------------
!
IF ( TURBN%LTURB_FLX .AND. TPFILE%LOPENED .AND. TURBN%CTURBDIM == '1DIM') THEN
  CALL GZ_W_M_PHY(D,PWM,PDZZ,ZWORK1)
  !$mnh_expand_array(JIJ=IIJB:IIJE,JK=1:IKT)
  ZFLXZ(:,:)= (2./3.) * PTKEM(:,:)                     &
     -ZCMFS*PLM(:,:)*SQRT(PTKEM(:,:))*ZWORK1(:,:)
  !$mnh_end_expand_array(JIJ=IIJB:IIJE,JK=1:IKT)
  ! to be tested &
  !   +XCMFB*(4./3.)*PLM(:,:,:)/SQRT(PTKEM(:,:,:))*PTP(:,:,:)
  ! stores the W variance
  TZFIELD = TFIELDMETADATA(      &
    CMNHNAME   = 'W_VVAR',       &
    CSTDNAME   = '',             &
    CLONGNAME  = 'W_VVAR',       &
    CUNITS     = 'm2 s-2',       &
    CDIR       = 'XY',           &
    CCOMMENT   = 'X_Y_Z_W_VVAR', &
    NGRID      = 1,              &
    NTYPE      = TYPEREAL,       &
    NDIMS      = 3,              &
    LTIMEDEP   = .TRUE.          )
  CALL IO_FIELD_WRITE_PHY(D,TPFILE,TZFIELD,ZFLXZ)
END IF
!
!----------------------------------------------------------------------------
!
IF (LHOOK) CALL DR_HOOK('TURB_VER_DYN_FLUX',1,ZHOOK_HANDLE)

CONTAINS
SUBROUTINE MY_UNPACK(PVARIN, PVAROUT) ! unpack the horizontal dimensions for io_field_write_user
IMPLICIT NONE
REAL, DIMENSION(D%NIT,D%NJT,D%NKT), INTENT(IN)   :: PVARIN
REAL, DIMENSION(D%NIT,D%NJT,D%NKT), INTENT(OUT)   :: PVAROUT

PVAROUT = PVARIN
END SUBROUTINE MY_UNPACK

END SUBROUTINE TURB_VER_DYN_FLUX

END MODULE MODE_TURB_VER_DYN_FLUX
