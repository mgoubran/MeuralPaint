#!/usr/bin/env bash

echo "suppressing TF logging"

# filter out info & warning logs
export TF_CPP_MIN_LOG_LEVEL=3
