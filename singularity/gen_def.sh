#!/bin/bash -e

SOURCE_DEF=$1
TARGET_DEF=$2
cp $SOURCE_DEF $TARGET_DEF

while true
do
	# min match
	HOST_ENV=$(awk 'match($0, /\$\{([^\}]*):HOST\}/, a){print a[1]}' $TARGET_DEF)
	if [ -z "$HOST_ENV" ]; then
		break
	fi

	for ENVNAME in ${HOST_ENV[@]}; do
		sed -i -r "s/\\$\{${ENVNAME}:HOST\}/${!ENVNAME}/g" $TARGET_DEF
	done
done
