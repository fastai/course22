#pragma once

#include <cuda.h>
#include <cupti_result.h>

#include <stddef.h>
#include <stdint.h>

namespace NV { namespace Cupti { namespace Checkpoint {

#ifdef __cplusplus
extern "C"
{
#endif

/**
 * \defgroup CUPTI_CHECKPOINT_API CUPTI Checkpoint API
 * Functions, types, and enums that implement the CUPTI Checkpoint API.
 * @{
 */

/**
 * \brief Specifies optimization options for a checkpoint, may be OR'd together to specify multiple options.
 */
typedef enum
{
    CUPTI_CHECKPOINT_OPT_NONE     = 0, //!< Default behavior
    CUPTI_CHECKPOINT_OPT_TRANSFER = 1, //!< Determine which mem blocks have changed, and only restore those. This optimization is cached, which means cuptiCheckpointRestore must always be called at the same point in the application when this option is enabled, or the result may be incorrect.
} CUpti_CheckpointOptimizations;

/**
 * \brief Configuration and handle for a CUPTI Checkpoint
 *
 * A CUptiCheckpoint object should be initialized with desired options prior to passing into any
 * CUPTI Checkpoint API function.  The first call into a Checkpoint API function will initialize internal
 * state based on these options.  Subsequent changes to these options will not have any effect.
 *
 * Checkpoint data is saved in device, host, and filesystem space.  There are options to reserve memory
 * at each level (device, host, filesystem) which are intended to allow a guarantee that a certain amount
 * of memory will remain free for use after the checkpoint is saved.
 * Note, however, that falling back to slower levels of memory (host, and then filesystem) to save the checkpoint
 * will result in performance degradation.
 * Currently, the filesystem limitation is not implemented.  Note that falling back to filesystem storage may
 * significantly impact the performance for saving and restoring a checkpoint.
 */
typedef struct
{
   size_t structSize;      //!< [in] Must be set to CUpti_Checkpoint_STRUCT_SIZE

   CUcontext ctx;          //!< [in] Set to context to save from, or will use current context if NULL

   size_t reserveDeviceMB; //!< [in] Restrict checkpoint from using last N MB of device memory (-1 = use no device memory)
   size_t reserveHostMB;   //!< [in] Restrict checkpoint from using last N MB of host memory (-1 = use no host memory)
   uint8_t allowOverwrite; //!< [in] Boolean, Allow checkpoint to save over existing checkpoint
   uint8_t optimizations;  //!< [in] Mask of CUpti_CheckpointOptimizations flags for this checkpoint

   void * pPriv;           //!< [in] Assign to NULL
} CUpti_Checkpoint;

#define CUpti_Checkpoint_STRUCT_SIZE  \
(offsetof(CUpti_Checkpoint, pPriv) +  \
sizeof(((CUpti_Checkpoint*)(nullptr))->pPriv))

#if defined(__GNUC__) && defined(CUPTI_LIB)
    #pragma GCC visibility push(default)
#endif

/**
 * \brief Initialize and save a checkpoint of the device state associated with the handle context
 *
 * Uses the handle options to configure and save a checkpoint of the device state associated with the specified context.
 *
 * \param handle A pointer to a CUpti_Checkpoint object
 *
 * \retval CUPTI_SUCCESS if a checkpoint was successfully initialized and saved
 * \retval CUPTI_ERROR_INVALID_PARAMETER if \p handle does not appear to refer to a valid CUpti_Checkpoint
 * \retval CUPTI_ERROR_INVALID_CONTEXT
 * \retval CUPTI_ERROR_INVALID_DEVICE if device associated with context is not compatible with checkpoint API
 * \retval CUPTI_ERROR_INVALID_OPERATION if Save is requested over an existing checkpoint, but \p allowOverwrite was not originally specified
 * \retval CUPTI_ERROR_OUT_OF_MEMORY if as configured, not enough backing storage space to save the checkpoint
 */
CUptiResult cuptiCheckpointSave(CUpti_Checkpoint * const handle);

/**
 * \brief Restore a checkpoint to the device associated with its context
 *
 * Restores device, pinned, and allocated memory to the state when the checkpoint was saved
 *
 * \param handle A pointer to a previously saved CUpti_Checkpoint object
 *
 * \retval CUTPI_SUCCESS if the checkpoint was successfully restored
 * \retval CUPTI_ERROR_NOT_INITIALIZED if the checkpoint was not previously initialized
 * \retval CUPTI_ERROR_INVALID_CONTEXT
 * \retval CUPTI_ERROR_INVALID_PARAMETER if the handle appears invalid
 * \retval CUPTI_ERROR_UNKNOWN if the restore or optimization operation fails
 */
CUptiResult cuptiCheckpointRestore(CUpti_Checkpoint * const handle);

/**
 * \brief Free the backing data for a checkpoint
 *
 * Frees all associated device, host memory and filesystem storage used for this context.
 * After freeing a handle, it may be re-used as if it was new - options may be re-configured and will
 * take effect on the next call to \p cuptiCheckpointSave.
 *
 * \param handle A pointer to a previously saved CUpti_Checkpoint object
 *
 * \retval CUPTI_SUCCESS if the handle was successfully freed
 * \retval CUPTI_ERROR_INVALID_PARAMETER if the handle was already freed or appears invalid
 * \retval CUPTI_ERROR_INVALID_CONTEXT if the context is no longer valid
 */
CUptiResult cuptiCheckpointFree(CUpti_Checkpoint * const handle);

#if defined(__GNUC__) && defined(CUPTI_LIB)
    #pragma GCC visibility pop
#endif

/**
 * @}
 */

#ifdef __cplusplus
}
#endif

// Exit namespace NV::Cupti::Checkpoint
}}}
