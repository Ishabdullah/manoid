#!/bin/bash
tail -n 0 -f marathon.log | grep --line-buffered -m 1 "\[Episode 985\]"
