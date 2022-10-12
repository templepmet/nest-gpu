#!/bin/bash -e

PROFILE_RANK=0
# PROFILE_RANK=ALL

if [ "$PROFILE_RANK" = "ALL" ]; then
	nsys profile --trace cuda,nvtx,mpi --output=report_rank$OMPI_COMM_WORLD_RANK --force-overwrite true \
	$@
elif [ "$PROFILE_RANK" = "$OMPI_COMM_WORLD_RANK" ]; then
	nsys profile --trace cuda,nvtx,mpi --output=report_rank$PROFILE_RANK --force-overwrite true \
	$@
elif [ -z "$OMPI_COMM_WORLD_RANK" ]; then
	nsys profile --trace cuda,nvtx,mpi --output=report --force-overwrite true \
	$@
else
	$@
fi
