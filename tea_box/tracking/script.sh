for i in {9775..9821}
do
    j=`expr $i - 9775`
    ln -s DSC_${i}.JPG LINK_`printf %04d ${j}`.JPG
done