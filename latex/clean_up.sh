# CLEAN UP
# Pass folder directory to the script
shopt -s extglob nocaseglob
cd $1
for file in !(@(*.sty|*.pdf|*.tex|*.sh|*.txss|*.xmpdata|*.bib|*.cls|*.bst|*.sty|*.eps)); do
    [[ -f "${file}" ]] && files+=( "${file}" )
done
(( ${#files[@]} )) && rm "${files[@]}" 
