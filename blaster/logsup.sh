#!/bin/sh
if [[ $# -lt 3 ]] ; then
  echo "$0 <emails-file> <output> [logfile_1] [logfile_2] ..."
  exit 0
fi

EMAILS_FILE=$1
shift
TARGET_FILE=$1
shift
LOG_FILE=$*
echo "Suppressing $LOG_FILE emails from $EMAILS_FILE to $TARGET_FILE"

SUP_FILE="/tmp/logsup.suppress"
cat $LOG_FILE | cut -d' ' -f2 > $SUP_FILE

SOURCE_FILE="/tmp/logsup.source"
cat $EMAILS_FILE | awk '{printf "S:jclick:6075:jclick:"} {print $1}' > $SOURCE_FILE

PRETARGET_FILE="/tmp/logsource.target"
sup -e $SUP_FILE $SOURCE_FILE $PRETARGET_FILE
cat $PRETARGET_FILE | cut -d: -f5 > $TARGET_FILE

set -x
wc -l $SOURCE_FILE
wc -l $SUP_FILE
wc -l $TARGET_FILE

