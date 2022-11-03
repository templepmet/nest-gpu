#!/bin/bash -e

PROFILE_RANK=$1
COMMAND=$@
EXE=${COMMAND/$1/}

if [ "$PROFILE_RANK" = "ALL" ]; then
	nsys profile --trace cuda,nvtx,mpi --output=report_rank$OMPI_COMM_WORLD_RANK --force-overwrite true \
	$EXE
elif [ "$PROFILE_RANK" = "$OMPI_COMM_WORLD_RANK" ]; then
	nsys profile --trace cuda,nvtx,mpi --output=report_rank$PROFILE_RANK --force-overwrite true \
	$EXE
elif [ -z "$OMPI_COMM_WORLD_RANK" ]; then
	nsys profile --trace cuda,nvtx,mpi --output=report --force-overwrite true \
	$EXE
else
	sleep 5
	$EXE
fi
