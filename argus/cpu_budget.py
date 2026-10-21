"""Bound CPU inference parallelism so background work leaves capture responsive."""


def configure_opencv():
    import cv2
    cv2.setNumThreads(1)


def ort_options():
    import onnxruntime as ort
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    # Idle inference sessions must not busy-spin beside camera/safety threads.
    options.add_session_config_entry('session.intra_op.allow_spinning', '0')
    options.add_session_config_entry('session.inter_op.allow_spinning', '0')
    return options
