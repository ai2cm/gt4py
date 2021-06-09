!------------------------------------------------------------*- Fortran -*-----
!
!                              S E R I A L B O X
!
! This file is distributed under terms of BSD license.
! See LICENSE.txt for more information.
!
!------------------------------------------------------------------------------
!
!+ This module contains the interface for serializing k-blocked or j-blocked fields
!
!------------------------------------------------------------------------------

MODULE utils_ppser_buff

!------------------------------------------------------------------------------
!
! Description:
!
!   This module contains subroutines which allow to serialize k-blocked
!   or j-blocked fields using internal buffering of the data before flushing them
!   off to serialbox. It uses buffering of the fields and fields are
!   automatically flushed once all data has been written.
!
! Current Code Owner: Vulcan Inc, Oliver Fuhrer
!  email:  oliverf@vulcan.com
!
! Code Description:
! Language: Fortran 90.
! Software Standards: "European Standards for Writing and
! Documenting Exchangeable Fortran 90 Code".
!============================================================================

USE iso_c_binding
USE m_serialize
USE utils_ppser
USE, INTRINSIC:: IEEE_ARITHMETIC, ONLY: IEEE_VALUE, IEEE_QUIET_NAN


IMPLICIT NONE

PUBLIC :: &
  fs_write_field_buff, finalize_buff

PRIVATE

  TYPE buff_type
    LOGICAL :: in_use = .FALSE.                   ! is this buffer in use?
    TYPE(C_PTR) :: serializer                     ! serializer object associated with buffer
    CHARACTER(LEN=256) :: savepoint_name
    CHARACTER(LEN=256) :: fieldname
    INTEGER :: buff_dim = 0                       ! dimension to be buffered (2=j, 3=k)
    INTEGER :: dim_i = 0, dim_j = 0, dim_k = 0    ! dimensions of 3d-field to be serialized
    INTEGER :: call_index = 0                     ! track multiple buffers for the same savepoint name
                                                  ! and field being filled in parallel
    LOGICAL :: has_minushalos, has_plushalos
    INTEGER :: minushalos(3), plushalos(3)
    INTEGER :: field_type = 0                     ! 0 = not used, 1 = int, 2 = r4, 3 = r8
    INTEGER, ALLOCATABLE :: buff_3d_i4(:,:,:)
    REAL(KIND=C_FLOAT), ALLOCATABLE :: buff_3d_r4(:,:,:)
    REAL(KIND=C_DOUBLE), ALLOCATABLE :: buff_3d_r8(:,:,:)
    LOGICAL, ALLOCATABLE :: ok(:)                 ! has this index been written?
  END TYPE buff_type

  INTEGER, PARAMETER :: max_buff = 9999           ! increase in case you get errors
  TYPE(buff_type) :: buff(max_buff)            ! array containing buffers

  ! overload interface for different types and dimensions
  INTERFACE fs_write_field_buff
      MODULE PROCEDURE fs_write_buff_3d_i4
      MODULE PROCEDURE fs_write_buff_3d_r4
      MODULE PROCEDURE fs_write_buff_3d_r8
  END INTERFACE

  LOGICAL :: first_call = .TRUE.                  ! used for initialization

  LOGICAL, PARAMETER :: debug = .FALSE.           ! get verbose messaging

CONTAINS

!============================================================================

! initialize buffering: makes sure all buffers are set to not in use
SUBROUTINE init_buff()
  IMPLICIT NONE

  INTEGER :: idx

  IF (debug) THEN
    WRITE(0,*) 'DEBUG init_buff'
  END IF

  first_call = .FALSE.

  DO idx = 1, max_buff
    buff(idx)%in_use = .FALSE.
    buff(idx)%fieldname = ""
    buff(idx)%savepoint_name = ""
    buff(idx)%call_index = 0
  END DO

END SUBROUTINE init_buff

!============================================================================

! finalize buffering: should be called once all buffers have been flushed
SUBROUTINE finalize_buff()
  IMPLICIT NONE

  INTEGER :: idx

  IF (debug) THEN
    WRITE(0,*) 'DEBUG finalize_buff'
  END IF

  DO idx = 1, max_buff
    IF (buff(idx)%in_use) THEN
      WRITE(0,*) 'ERROR in utils_ppser_buff: finalize called before all buffers have been flushed'
      WRITE(0,*) 'savepoint_name: ', TRIM(buff(idx)%savepoint_name)
      WRITE(0,*) 'name of partially empty field: ', TRIM(buff(idx)%fieldname)
      WRITE(0,*) 'vertical indices filled boolean: ', buff(idx)%ok(:)
      STOP
    END IF
    buff(idx)%fieldname = ""
    buff(idx)%savepoint_name = ""
  END DO

  first_call = .TRUE.

END SUBROUTINE finalize_buff

!============================================================================

! overloads fs_write_buff: version for r8 floats and 3d fields
SUBROUTINE fs_write_buff_3d_r8(serializer, savepoint, fieldname, field, &
                               buff_dim, index, index_size, mode, minushalos, plushalos)
  IMPLICIT NONE

  TYPE(t_serializer), TARGET, INTENT(IN)  :: serializer
  TYPE(t_savepoint), TARGET, INTENT(IN)   :: savepoint
  CHARACTER(LEN=*), INTENT(IN)            :: fieldname
  REAL(KIND=C_DOUBLE), INTENT(IN), TARGET :: field(:,:)
  INTEGER, INTENT(IN)                     :: buff_dim, index, index_size, mode
  INTEGER, INTENT(IN), OPTIONAL           :: minushalos(3), plushalos(3)

  ! local vars
  INTEGER :: buff_id = 0
  INTEGER :: field_type = 3

  ! do nothing in case serialization is switched off
  IF (.NOT. (fs_is_serialization_on())) THEN
    RETURN
  ENDIF

  ! find buff_id and check if a buffer slot was found
  call setup_buffer(buff_id, serializer, savepoint, fieldname, field_type, &
                    SIZE(field,1), SIZE(field,2), buff_dim, index_size, index, &
                    mode, minushalos, plushalos)

  ! store data
  IF (debug) THEN
    WRITE(0,*) 'DEBUG fs_write_buff_3d_r8: store data'
  END IF
  IF (buff_dim == 1) THEN
    buff(buff_id)%buff_3d_r8(index,:,:) = field(:,:)
  ELSE IF (buff_dim == 2) THEN
    buff(buff_id)%buff_3d_r8(:,index,:) = field(:,:)
  ELSE IF (buff_dim == 3) THEN
    buff(buff_id)%buff_3d_r8(:,:,index) = field(:,:)
  END IF
  buff(buff_id)%ok(index) = .TRUE.

  ! write if we are complete
  IF (ALL(buff(buff_id)%ok(:))) THEN
    IF (debug) THEN
      WRITE(0,*) 'DEBUG fs_write_buff_3d_r8: flush data'
    END IF
    IF (buff(buff_id)%has_minushalos) THEN
      IF (buff(buff_id)%has_plushalos) THEN
        CALL fs_write_field(serializer, savepoint, fieldname, buff(buff_id)%buff_3d_r8, &
          minushalos=buff(buff_id)%minushalos, plushalos=buff(buff_id)%plushalos)
      ELSE
        CALL fs_write_field(serializer, savepoint, fieldname, buff(buff_id)%buff_3d_r8, &
          minushalos=buff(buff_id)%minushalos)
      END IF
    ELSE
      IF (buff(buff_id)%has_plushalos) THEN
        CALL fs_write_field(serializer, savepoint, fieldname, buff(buff_id)%buff_3d_r8, &
          plushalos=buff(buff_id)%plushalos)
      ELSE
        CALL fs_write_field(serializer, savepoint, fieldname, buff(buff_id)%buff_3d_r8)
      END IF
    END IF

    CALL destroy_buff(buff_id)
  END IF

END SUBROUTINE fs_write_buff_3d_r8

!============================================================================

! overloads fs_write_buff: version for r4 floats and 3d fields
SUBROUTINE fs_write_buff_3d_r4(serializer, savepoint, fieldname, field, &
                               buff_dim, index, index_size, mode, minushalos, plushalos)
  IMPLICIT NONE

  TYPE(t_serializer), TARGET, INTENT(IN)  :: serializer
  TYPE(t_savepoint), TARGET, INTENT(IN)   :: savepoint
  CHARACTER(LEN=*), INTENT(IN)            :: fieldname
  REAL(KIND=C_FLOAT), INTENT(IN), TARGET  :: field(:,:)
  INTEGER, INTENT(IN)                     :: buff_dim, index, index_size, mode
  INTEGER, INTENT(IN), OPTIONAL           :: minushalos(3), plushalos(3)

  ! local vars
  INTEGER :: buff_id = 0
  INTEGER :: field_type = 2

  ! do nothing in case serialization is switched off
  IF (.NOT. (fs_is_serialization_on())) THEN
    RETURN
  ENDIF

  ! find buff_id and check if a buffer slot was found
  call setup_buffer(buff_id, serializer, savepoint, fieldname, field_type, &
                    SIZE(field,1), SIZE(field,2), buff_dim, index_size, index, &
                    mode, minushalos, plushalos)

  ! store data
  IF (debug) THEN
    WRITE(0,*) 'DEBUG fs_write_buff_3d_r4: store data'
  END IF
  IF (buff_dim == 1) THEN
    buff(buff_id)%buff_3d_r4(index,:,:) = field(:,:)
  ELSE IF (buff_dim == 2) THEN
    buff(buff_id)%buff_3d_r4(:,index,:) = field(:,:)
  ELSE IF (buff_dim == 3) THEN
    buff(buff_id)%buff_3d_r4(:,:,index) = field(:,:)
  END IF
  buff(buff_id)%ok(index) = .TRUE.

  ! write if we are complete
  IF (ALL(buff(buff_id)%ok(:))) THEN
    IF (debug) THEN
      WRITE(0,*) 'DEBUG fs_write_buff_3d_r4: flush data'
    END IF
    IF (buff(buff_id)%has_minushalos) THEN
      IF (buff(buff_id)%has_plushalos) THEN
        CALL fs_write_field(serializer, savepoint, fieldname, buff(buff_id)%buff_3d_r4, &
          minushalos=buff(buff_id)%minushalos, plushalos=buff(buff_id)%plushalos)
      ELSE
        CALL fs_write_field(serializer, savepoint, fieldname, buff(buff_id)%buff_3d_r4, &
          minushalos=buff(buff_id)%minushalos)
      END IF
    ELSE
      IF (buff(buff_id)%has_plushalos) THEN
        CALL fs_write_field(serializer, savepoint, fieldname, buff(buff_id)%buff_3d_r4, &
          plushalos=buff(buff_id)%plushalos)
      ELSE
        CALL fs_write_field(serializer, savepoint, fieldname, buff(buff_id)%buff_3d_r4)
      END IF
    END IF

    CALL destroy_buff(buff_id)
  END IF

END SUBROUTINE fs_write_buff_3d_r4

!============================================================================

! overloads fs_write_buff: version for i4 integers and 3d fields
SUBROUTINE fs_write_buff_3d_i4(serializer, savepoint, fieldname, field, &
                               buff_dim, index, index_size, mode, minushalos, plushalos)
  IMPLICIT NONE

  TYPE(t_serializer), TARGET, INTENT(IN)  :: serializer
  TYPE(t_savepoint), TARGET, INTENT(IN)   :: savepoint
  CHARACTER(LEN=*), INTENT(IN)            :: fieldname
  INTEGER, INTENT(IN), TARGET             :: field(:,:)
  INTEGER, INTENT(IN)                     :: buff_dim, index, index_size, mode
  INTEGER, INTENT(IN), OPTIONAL           :: minushalos(3), plushalos(3)

  ! local vars
  INTEGER :: buff_id = 0
  INTEGER :: field_type = 1

  ! do nothing in case serialization is switched off
  IF (.NOT. (fs_is_serialization_on())) THEN
    RETURN
  ENDIF

  ! find buff_id and check if a buffer slot was found
  call setup_buffer(buff_id, serializer, savepoint, fieldname, field_type, &
                    SIZE(field,1), SIZE(field,2), buff_dim, index_size, index, &
                    mode, minushalos, plushalos)

  ! store data
  IF (debug) THEN
    WRITE(0,*) 'DEBUG fs_write_buff_3d_i4: store data'
  END IF
  IF (buff_dim == 1) THEN
    buff(buff_id)%buff_3d_i4(index,:,:) = field(:,:)
  ELSE IF (buff_dim == 2) THEN
    buff(buff_id)%buff_3d_i4(:,index,:) = field(:,:)
  ELSE IF (buff_dim == 3) THEN
    buff(buff_id)%buff_3d_i4(:,:,index) = field(:,:)
  END IF
  buff(buff_id)%ok(index) = .TRUE.

  ! write if we are complete
  IF (ALL(buff(buff_id)%ok(:))) THEN
    IF (debug) THEN
      WRITE(0,*) 'DEBUG fs_write_buff_3d_i4: flush data'
    END IF
    IF (buff(buff_id)%has_minushalos) THEN
      IF (buff(buff_id)%has_plushalos) THEN
        CALL fs_write_field(serializer, savepoint, fieldname, buff(buff_id)%buff_3d_i4, &
          minushalos=buff(buff_id)%minushalos, plushalos=buff(buff_id)%plushalos)
      ELSE
        CALL fs_write_field(serializer, savepoint, fieldname, buff(buff_id)%buff_3d_i4, &
          minushalos=buff(buff_id)%minushalos)
      END IF
    ELSE
      IF (buff(buff_id)%has_plushalos) THEN
        CALL fs_write_field(serializer, savepoint, fieldname, buff(buff_id)%buff_3d_i4, &
          plushalos=buff(buff_id)%plushalos)
      ELSE
        CALL fs_write_field(serializer, savepoint, fieldname, buff(buff_id)%buff_3d_i4)
      END IF
    END IF

    CALL destroy_buff(buff_id)
  END IF

END SUBROUTINE fs_write_buff_3d_i4

!============================================================================

! checks if a buffer exists for this fields and if yes, checks consistency with
! current request. if not, it creates a new buffer.
SUBROUTINE setup_buffer(buff_id, serializer, savepoint, fieldname, field_type, &
                        size_dim1, size_dim2, buff_dim, index_size, index, mode, &
                        minushalos, plushalos)
  IMPLICIT NONE

  TYPE(t_serializer), TARGET, INTENT(IN)  :: serializer
  TYPE(t_savepoint), TARGET, INTENT(IN)   :: savepoint
  CHARACTER(LEN=*), INTENT(IN)            :: fieldname
  INTEGER, INTENT(IN)                     :: buff_dim, index, index_size, mode, size_dim1, size_dim2, field_type
  INTEGER, INTENT(IN), OPTIONAL           :: minushalos(3), plushalos(3)
  INTEGER, INTENT(OUT)                    :: buff_id

  ! local vars
  INTEGER :: call_index = 0
  INTEGER :: i

  IF (debug) THEN
    WRITE(0,*) 'DEBUG setup_buffer: savepoint=', TRIM(savepoint%savepoint_name)
    WRITE(0,*) 'DEBUG setup_buffer: fieldname=', TRIM(fieldname)
    WRITE(0,*) 'DEBUG setup_buffer: buff_dim=', buff_dim
    WRITE(0,*) 'DEBUG setup_buffer: index=', index
    WRITE(0,*) 'DEBUG setup_buffer: index_size=', index_size
  END IF

  ! ppser mode numbers do not align with m_serialize constants....
  IF ( mode /= PPSER_MODE_WRITE ) THEN
    WRITE(0,*) 'ERROR, can only use buffer in write mode'
    STOP
  END IF

  ! initialize if this is the first call
  IF ( first_call ) THEN
    CALL init_buff()
  END IF

  ! find ID if it already exists
  CALL find_buff_id(fieldname, savepoint, buff_dim, index, buff_id, call_index)
  IF (debug) THEN
    WRITE(0,*) 'DEBUG fs_write_buff_3d_r8: find buff_id=', buff_id
  END IF

  ! check if a buffer slot was found
  IF (buff_id == 0) THEN
    ! no, so create a new buffer
    CALL get_free_buff_id(buff_id)
    IF (debug) THEN
      WRITE(0,*) 'DEBUG fs_write_buff_3d: buff_id=', buff_id
    END IF
    CALL create_buff(buff_id, serializer, savepoint, fieldname, field_type, &
                     size_dim1, size_dim2, buff_dim, index_size, call_index, minushalos, plushalos)
  ELSE
    ! yes, so check for consistency of current request with stored metadata
    CALL check_buff(buff_id, serializer, savepoint, fieldname, field_type, &
                    size_dim1, size_dim2, buff_dim, index_size, index, minushalos, plushalos)
  END IF

 END SUBROUTINE setup_buffer

!============================================================================


! create a new buffer (allocate memory, store metadata)
SUBROUTINE create_buff(buff_id, serializer, savepoint, fieldname, field_type, &
                       size_dim1, size_dim2, buff_dim, index_size, call_index, minushalos, plushalos)
  IMPLICIT NONE

  INTEGER, INTENT(IN)                     :: buff_id
  TYPE(t_serializer), TARGET, INTENT(IN)  :: serializer
  TYPE(t_savepoint), TARGET, INTENT(IN)   :: savepoint
  CHARACTER(LEN=*), INTENT(IN)            :: fieldname
  INTEGER, INTENT(IN)                     :: field_type, buff_dim
  INTEGER, INTENT(IN)                     :: size_dim1, size_dim2, index_size
  INTEGER, INTENT(IN)                     :: call_index
  INTEGER, INTENT(IN), OPTIONAL           :: minushalos(3), plushalos(3)

  INTEGER :: dim_i, dim_j, dim_k

  ! debug information
  IF (debug) THEN
    WRITE(0,*) 'DEBUG create_buff: buff_id=', buff_id
    WRITE(0,*) 'DEBUG create_buff: savepoint=', TRIM(savepoint%savepoint_name)
    WRITE(0,*) 'DEBUG create_buff: fieldname=', TRIM(fieldname)
    WRITE(0,*) 'DEBUG create_buff: field_type=', field_type
    WRITE(0,*) 'DEBUG create_buff: size_dim1,size_dim2,index_size=', size_dim1, size_dim2, index_size
  END IF

  ! security check
  IF (buff_id < 1 .OR. buff_id > max_buff) THEN
    WRITE(0,*) 'ERROR in utils_ppser_buff: illegal buff_id encountered'
    STOP
  END IF
  IF (buff(buff_id)%in_use) THEN
    WRITE(0,*) 'ERROR in utils_ppser_buff: create called for buffer already in use'
    STOP
  END IF

  ! get dimensions
  IF (buff_dim == 1) THEN
    dim_i = index_size
    dim_j = size_dim1
    dim_k = size_dim2
  ELSE IF (buff_dim ==2) THEN
    dim_i = size_dim1
    dim_j = index_size
    dim_k = size_dim2
  ELSE IF (buff_dim ==3) THEN
    dim_i = size_dim1
    dim_j = size_dim2
    dim_k = index_size
  END IF

  ! store metadata
  buff(buff_id)%in_use = .TRUE.
  buff(buff_id)%serializer = serializer%serializer_ptr
  buff(buff_id)%call_index = call_index
  buff(buff_id)%savepoint_name = TRIM(savepoint%savepoint_name)
  buff(buff_id)%fieldname = TRIM(fieldname)
  buff(buff_id)%buff_dim = buff_dim
  buff(buff_id)%dim_i = dim_i
  buff(buff_id)%dim_j = dim_j
  buff(buff_id)%dim_k = dim_k
  IF (PRESENT(minushalos)) THEN
    buff(buff_id)%minushalos = minushalos
    buff(buff_id)%has_minushalos = .TRUE.
  ELSE
    buff(buff_id)%has_minushalos = .FALSE.
  ENDIF
  IF (PRESENT(plushalos)) THEN
    buff(buff_id)%plushalos = plushalos
    buff(buff_id)%has_plushalos = .TRUE.
  ELSE
    buff(buff_id)%has_plushalos = .FALSE.
  ENDIF
  buff(buff_id)%field_type = field_type

  ! allocate memory
  SELECT CASE (field_type)
    CASE(1)
      ALLOCATE(buff(buff_id)%buff_3d_i4(dim_i, dim_j, dim_k))
      buff(buff_id)%buff_3d_i4 = HUGE(1)
    CASE(2)
      ALLOCATE(buff(buff_id)%buff_3d_r4(dim_i, dim_j, dim_k))
      buff(buff_id)%buff_3d_r4 = IEEE_VALUE(1.e0, IEEE_QUIET_NAN)
    CASE(3)
      ALLOCATE(buff(buff_id)%buff_3d_r8(dim_i, dim_j, dim_k))
      buff(buff_id)%buff_3d_r8 = IEEE_VALUE(1.d0, IEEE_QUIET_NAN)
    CASE DEFAULT
      WRITE(0,*) 'ERROR in utils_ppser_buff: unsupported field_type encountered'
  END SELECT
  ALLOCATE(buff(buff_id)%ok(index_size))

  ! make sure all buffer indices are marked as unwritten
  buff(buff_id)%ok(:) = .FALSE.

END SUBROUTINE create_buff

!============================================================================

! release a buffer (release memory, reset metadata)
SUBROUTINE destroy_buff(buff_id)
  IMPLICIT NONE

  INTEGER, INTENT(IN)                     :: buff_id
  INTEGER                                 :: idx
  ! debug information
  IF (debug) THEN
    WRITE(0,*) 'DEBUG destroy_buff: buff_id=', buff_id
    WRITE(0,*) 'DEBUG destroy_buff: savepoint=', TRIM(buff(buff_id)%savepoint_name)
    WRITE(0,*) 'DEBUG destroy_buff: fieldname=', TRIM(buff(buff_id)%fieldname)
  END IF

  ! security check
  IF (buff_id < 1 .OR. buff_id > max_buff) THEN
    WRITE(0,*) 'ERROR in utils_ppser_buff: illegal buff_id encountered'
    STOP
  END IF
  IF (.NOT. buff(buff_id)%in_use) THEN
    WRITE(0,*) 'ERROR in utils_ppser_buff: destroy called for buffer not in use'
    STOP
  END IF

  ! update the call_index of the rest of the related buffers
  DO idx = 1, max_buff
    IF (idx /= buff_id .and. buff(idx)%in_use) THEN
      IF (TRIM(buff(buff_id)%fieldname) == TRIM(buff(idx)%fieldname)) THEN
        IF (TRIM(buff(buff_id)%savepoint_name) == TRIM(buff(idx)%savepoint_name)) THEN
          IF (buff(buff_id)%buff_dim == buff(idx)%buff_dim) THEN
            ! This should not be needed, calls should stay in order...
            IF (buff(idx)%call_index >  buff(buff_id)%call_index) THEN
              buff(idx)%call_index = buff(idx)%call_index - 1
            END IF
          END IF
        END IF
      END IF
    END IF
  END DO

  ! release memory
  SELECT CASE (buff(buff_id)%field_type)
    CASE(1)
      DEALLOCATE(buff(buff_id)%buff_3d_i4)
    CASE(2)
      DEALLOCATE(buff(buff_id)%buff_3d_r4)
    CASE(3)
      DEALLOCATE(buff(buff_id)%buff_3d_r8)
    CASE DEFAULT
      WRITE(0,*) 'ERROR in utils_ppser_buff: unsupported field_type encountered in destroy'
  END SELECT
  DEALLOCATE(buff(buff_id)%ok)

  ! reset metadata
  buff(buff_id)%in_use = .FALSE.
  buff(buff_id)%serializer = C_NULL_PTR
  buff(buff_id)%savepoint_name = ""
  buff(buff_id)%call_index = 0
  buff(buff_id)%fieldname = ""
  buff(buff_id)%field_type = 0
  buff(buff_id)%buff_dim = 0
  buff(buff_id)%dim_i = 0
  buff(buff_id)%dim_j = 0
  buff(buff_id)%dim_k = 0
  IF (buff(buff_id)%has_minushalos) THEN
    buff(buff_id)%minushalos = (/0, 0, 0/)
  ENDIF
  buff(buff_id)%has_minushalos = .FALSE.
  IF (buff(buff_id)%has_plushalos) THEN
    buff(buff_id)%plushalos = (/0, 0, 0/)
  ENDIF
  buff(buff_id)%has_plushalos = .FALSE.

END SUBROUTINE destroy_buff

!============================================================================

! check consistency of current request with metadata stored in buffer
SUBROUTINE check_buff(buff_id, serializer, savepoint, fieldname, field_type, &
                      size_dim1, size_dim2, buff_dim, index_size, index, minushalos, plushalos)
  IMPLICIT NONE

  INTEGER, INTENT(IN)                     :: buff_id
  TYPE(t_serializer), TARGET, INTENT(IN)  :: serializer
  TYPE(t_savepoint), TARGET, INTENT(IN)   :: savepoint
  CHARACTER(LEN=*), INTENT(IN)            :: fieldname
  INTEGER, INTENT(IN)                     :: field_type, buff_dim
  INTEGER, INTENT(IN)                     :: size_dim1, size_dim2, index_size, index
  INTEGER, INTENT(IN), OPTIONAL           :: minushalos(3), plushalos(3)

  INTEGER :: dim_i, dim_j, dim_k

  ! debug information
  IF (debug) THEN
    WRITE(0,*) 'DEBUG check_buff: buff_id=', buff_id
    WRITE(0,*) 'DEBUG check_buff: savepoint=', TRIM(savepoint%savepoint_name)
    WRITE(0,*) 'DEBUG check_buff: fieldname=', TRIM(fieldname)
    WRITE(0,*) 'DEBUG check_buff: field_type=', field_type
    WRITE(0,*) 'DEBUG check_buff: buff_dim=', buff_dim
    WRITE(0,*) 'DEBUG check_buff: size_dim1,size_dim2,index_size=', size_dim1, size_dim2, index_size
    WRITE(0,*) 'DEBUG check_buff: index=', index
  END IF

  ! security check
  IF (buff_id < 1 .OR. buff_id > max_buff) THEN
    WRITE(0,*) 'ERROR in utils_ppser_buff: illegal buff_id encountered'
    STOP
  END IF
  IF (.NOT. buff(buff_id)%in_use) THEN
    WRITE(0,*) 'ERROR in utils_ppser_buff: check called for buffer not in use'
    STOP
  END IF

  ! get dimensions
  IF (buff_dim == 1) THEN
    dim_i = index_size
    dim_j = size_dim1
    dim_k = size_dim2
  ELSE IF (buff_dim ==2) THEN
    dim_i = size_dim1
    dim_j = index_size
    dim_k = size_dim2
  ELSE IF (buff_dim ==3) THEN
    dim_i = size_dim1
    dim_j = size_dim2
    dim_k = index_size
  END IF

  ! check consistency
  IF (.NOT. (TRIM(buff(buff_id)%fieldname) == TRIM(fieldname))) THEN
    WRITE(0,*) 'ERROR in utils_ppser_buff: inconsistent name encountered'
    STOP
  END IF
  IF (.NOT. (C_ASSOCIATED(buff(buff_id)%serializer, serializer%serializer_ptr))) THEN
    WRITE(0,*) 'ERROR in utils_ppser_buff: write called for same field but different serializer'
    STOP
  END IF
  IF (.NOT. (TRIM(buff(buff_id)%savepoint_name) == TRIM(savepoint%savepoint_name))) THEN
    WRITE(0,*) 'ERROR in utils_ppser_buff: write called for same field but different savepoint'
    STOP
  END IF
  IF (buff(buff_id)%buff_dim /= buff_dim) THEN
    WRITE(0,*) 'ERROR in utils_ppser_buff: write called with inconsistent buffer dimension'
    STOP
  END IF
  IF (ANY( (/buff(buff_id)%dim_i, buff(buff_id)%dim_j, buff(buff_id)%dim_k/) /= (/dim_i, dim_j, dim_k/) )) THEN
    WRITE(0,*) 'ERROR in utils_ppser_buff: write called with inconsistent dimensions'
    STOP
  END IF
  IF ((index < 1) .OR. (index > index_size)) THEN
    WRITE(0,*) 'ERROR in utils_ppser_buff: out of bound index encountered'
    STOP
  END IF
  IF (buff(buff_id)%has_minushalos .AND. PRESENT(minushalos)) THEN
    IF (ANY(buff(buff_id)%minushalos /= minushalos)) THEN
      WRITE(0,*) 'ERROR in utils_ppser_buff: inconsistent minushalos encountered'
      STOP
    END IF
  END IF
  IF (buff(buff_id)%has_plushalos .AND. PRESENT(plushalos)) THEN
    IF (ANY(buff(buff_id)%plushalos /= plushalos)) THEN
      WRITE(0,*) 'ERROR in utils_ppser_buff: inconsistent plushalos encountered'
      STOP
    END IF
  END IF
  IF (buff(buff_id)%field_type /= field_type) THEN
    WRITE(0,*) 'ERROR in utils_ppser_buff: write with inconsistent field_type encountered'
    STOP
  END IF
  ! Should be redundant, but doesn't hurt to recheck
  IF (buff(buff_id)%ok(index)) THEN
    WRITE(0,*) 'ERROR in utils_ppser_buff: index already written'
    STOP
  END IF

END SUBROUTINE check_buff

!============================================================================


! find the ID of a buffer given name of field and savepoint
SUBROUTINE find_buff_id(fieldname, savepoint, buff_dim, index, buff_id, call_index)
  IMPLICIT NONE

  CHARACTER(LEN=*), INTENT(IN)            :: fieldname
  TYPE(t_savepoint), TARGET, INTENT(IN)   :: savepoint
  INTEGER, INTENT(IN)                     :: index, buff_dim
  INTEGER, INTENT(OUT)                    :: buff_id, call_index

  ! local vars
  INTEGER :: idx

  buff_id = 0
  call_index = 0
  IF (debug) THEN
    WRITE(0,*) 'DEBUG find_buff_id: fieldname=', TRIM(fieldname), ' savepoint=', TRIM(savepoint%savepoint_name)
  END IF

  DO idx = 1, max_buff
    IF (buff(idx)%in_use) THEN
      IF (TRIM(fieldname) == TRIM(buff(idx)%fieldname)) THEN
        IF (TRIM(savepoint%savepoint_name) == TRIM(buff(idx)%savepoint_name)) THEN
          IF (buff(idx)%buff_dim == buff_dim) THEN
            IF (debug) THEN
              WRITE(0, *) 'DEBUG found name match at ', idx, 'index=', index, buff(idx)%ok(index)
            END IF
            IF (buff(idx)%ok(index)) THEN
              ! The index for this buffer has already been filled, keep looking for another with the same name
              call_index = call_index + 1
            ELSE
              IF (debug) THEN
                WRITE(0, *) 'DEBUG found buff_id', idx, ', counted similar buffers: ', call_index
              END IF
              buff_id = idx
              EXIT
            END IF
          END IF
        END IF
      END IF
    END IF
  END DO

  IF (debug) THEN
    IF (buff_id == 0) THEN
      WRITE(0,*) 'DEBUG find_buff_id: no buff found, call_index=', call_index
    ELSE
      WRITE(0,*) 'DEBUG find_buff_id: found buff_id=', buff_id, ' call_index=', call_index
    END IF
  END IF

END SUBROUTINE find_buff_id

!============================================================================

! find a free buffer ID
SUBROUTINE get_free_buff_id(buff_id)
  IMPLICIT NONE

  INTEGER, INTENT(OUT) :: buff_id

  ! local vars
  INTEGER :: idx

  buff_id = 0

  ! find a free index
  DO idx = 1, max_buff
    IF (.NOT. buff(idx)%in_use) THEN
      buff_id = idx
      EXIT
    END IF
  END DO

  ! abort if no free index has been found
  IF (idx > max_buff) THEN
    WRITE(0,*) 'ERROR in utils_ppser_buff: no more free buffers (increase max_buff)'
    STOP
  END IF

END SUBROUTINE get_free_buff_id

!============================================================================

END MODULE utils_ppser_buff
