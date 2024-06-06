from __future__ import print_function

import numpy as np
import resampy  # pylint: disable=import-error
import tensorflow.compat.v1 as tf
import pandas as pd
import sys
import platform
import psutil

# Collecting system information
system_info = {
    "Component": ["System", "Release", "Machine", "Python Version", "pandas", "numpy", "tensorflow", "Total RAM (GB)", "Number of GPUs"],
    "Details": [
        platform.system(),
        platform.release(),
        platform.machine(),
        sys.version.split(" ")[0],
        pd.__version__,
        np.__version__,
        tf.__version__,
        str(round(psutil.virtual_memory().total / (1024 ** 3), 1)),
        str(len([gpu for gpu in tf.config.experimental.list_physical_devices('GPU') if 'eGPU' in gpu.device_type]))
    ]
}

# Creating a DataFrame for display
df_system_info = pd.DataFrame(system_info)

# Printing the DataFrame
print(df_system_info.to_string(index=False))
