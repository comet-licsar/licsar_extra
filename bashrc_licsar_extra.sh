#!/bin/bash
## Automatically detect LiCSAR Extra path based on this script's location

export LiCSAR_extra_path="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

## Remove any previous occurrences of the default LiCSAR Extra path
export PYTHONPATH=$(echo "$PYTHONPATH" | awk -v RS=: -v ORS=: '$0 !~ "/gws/smf/j04/nceo_geohazards/software/licsar_extra/python"' | sed 's/:$//')

## Now, prepend your own local path
export PYTHONPATH="$LiCSAR_extra_path/python:$PYTHONPATH"

# ## Confirm change
# echo "Custom LiCSAR Extra path applied: $LiCSAR_extra_path/python"
# echo $PYTHONPATH | tr ':' '\n' | grep licsar_extra