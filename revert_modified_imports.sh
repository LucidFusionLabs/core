# remove modified imports files
svn diff imports | grep "Index: " | cut -d' ' -f2 | xargs svn revert

