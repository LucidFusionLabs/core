# remove modified imports files
git diff --name-only imports | xargs git checkout --
