# unpack compressed files
cat Q0_public/*.tgz | tar -izx -C Q0_public
cat Q1_public/*.tgz | tar -izx -C Q1_public
cat Q2_public/*.tgz | tar -izx -C Q2_public
cat Q3_public/*.tgz | tar -izx -C Q3_public
cat Q4_public/*.tgz | tar -izx -C Q4_public
cat Q5_public/*.tgz | tar -izx -C Q5_public
cat Q6_public/*.tgz | tar -izx -C Q6_public
cat Q7_public/*.tgz | tar -izx -C Q7_public
cat Q8_public/*.tgz | tar -izx -C Q8_public
cat Q9_public/*.tgz | tar -izx -C Q9_public
cat Q10_public/*.tgz | tar -izx -C Q10_public
cat Q11_public/*.tgz | tar -izx -C Q11_public
cat Q12_public/*.tgz | tar -izx -C Q12_public
cat Q13_public/*.tgz | tar -izx -C Q13_public
cat Q14_public/*.tgz | tar -izx -C Q14_public
cat Q15_public/*.tgz | tar -izx -C Q15_public
cat Q16_public/*.tgz | tar -izx -C Q16_public
cat Q17_public/*.tgz | tar -izx -C Q17_public

# remove compressed files to save disk space
rm Q*_public/*.tgz