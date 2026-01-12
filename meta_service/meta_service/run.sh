#!/bin/bash

DB_PATH_BASE=$(pwd)
SERVER_IP="127.0.0.1"
START_PORT=7877
GRPC_START_PORT=7977
PARTY_COUNT=3

for ((i=0; i<$PARTY_COUNT; i++)); do
    DB_PATH=${DB_PATH_BASE}/storage/node$i
    SERVER_PORT=$(($START_PORT + $i))
    GRPC_SERVER_PORT=$(($GRPC_START_PORT + $i))
    COLLABORATE=""

    for ((j=0; j<$PARTY_COUNT; j++)); do
        if [ $i != $j ]; then
            COLLABORATE+="http://${SERVER_IP}:$(($START_PORT + $j))/,"
        fi
    done

    java -jar fusion-simple.jar \
        --server.port=$SERVER_PORT \
        --grpc.server.port=$GRPC_SERVER_PORT \
        --db.path=$DB_PATH \
        --collaborate=${COLLABORATE%?} \
        &> meta_log$i &
done
