library(natverse)
library(tidyverse)
library(fafbseg)
library(googlesheets4)
library(glue)
library(cowplot)
library(plotly)

# rm(list = ls())

### Checking that all works with the fafbseq package -------------------------
# dr_fafbseg()

### some functions -----------------------------------------------------------

## function that takes segment(s) as input and returns flywire URL
flywire_url_func <- function(segment_id){
  fw_url=with_segmentation('flywire', getOption('fafbseg.sampleurl'))
  ngl_segments(fw_url)
  fw_sc=ngl_decode_scene(fw_url)
  fw_sc$layers[[2]]$segments= segment_id
  browseURL(ngl_encode_url(fw_sc))
}


### "google_sheet" table --------------------------------------------------

## Read the "google_sheet" table
ss_str = "1RNVFgC8gjAk-9aJY7dtUZdIhUcVrd1b0cHpBVgAR9e8" # silies_neurons
ss_str = "1CLETHPvpeFm2AZeuOYEQ7GNSVygDfgjPQ_56FY7JeNY" # T4 proofreadings dorsal
google_sheet = read_sheet(ss_str)


## Update table (segment IDs)
{
  start_time <- Sys.time()
  google_sheet= google_sheet%>% separate(XYZ, c("voxel_raw_x","voxel_raw_y","voxel_raw_z"), sep = ",",remove = F)
  google_sheet$name = glue('Putative_{google_sheet$type}_{google_sheet$voxel_raw_x}_{google_sheet$voxel_raw_y}_{google_sheet$voxel_raw_z}')
  google_sheet$seg_id = flywire_xyz2id(matrix(c(as.numeric(google_sheet$voxel_raw_x),
                                                       as.numeric(google_sheet$voxel_raw_y),
                                                       as.numeric(google_sheet$voxel_raw_z)),
                                                     ncol = 3,
                                                     byrow = FALSE,
                                                     dimnames = list(NULL, c("X","Y","Z"))),
                                              rawcoords = TRUE,
                                              fast_root = TRUE)
  end_time <- Sys.time()
  run_time <- end_time - start_time
  run_time
}

## check for duplications (should return empty dataframe)
google_sheet %>% group_by(seg_id) %>% filter(n()>1) %>%  arrange(seg_id,type,hemisphere,name)

## remove duplications and arrange neurons according to type, hemisphere & name.
google_sheet = google_sheet %>% 
  distinct(seg_id, .keep_all = TRUE) %>% 
  arrange(type,hemisphere,name)

## Write back the "google_sheet" table
# sheet_write(google_sheet, ss=ss_str, sheet = "Sheet1")


### Loading data ------------------------------------------------------------

load("C:/Users/sebas/SiliesLab/Connectomics data/FlyWire/RData/Mi1_syn.RData")
load("C:/Users/sebas/SiliesLab/Connectomics data/FlyWire/RData/Mi1_ar.RData")


### Code --------------------------------------------------------------------

## Selecting columns

Mi1_center_ii = c(411, 438, 551, 616)

Mi1_st_ii = c(211, 347, 403, 470, 499,
              529, 550, 615, 673, 759, 
              768, 769, Mi1_center_ii)

Mi1_st_outer_ii =  c(45,  52,  56,  63,  71,
                     75,  81,  86, 105, 113, 
                     115, 122, 129, 140, 148, 
                     171, 178, 196, 201, 208, 
                     212, 218, 220, 221, 222, 
                     223, 229, 241, 246, 249, 
                     261, 277, 284, 295, 301, 
                     309, 320, 323, 329, 344, 
                     353, 362, 377, 378, 391, 
                     402, 404, 406, 407, 416, 
                     417, 420, 442, 449, 453,
                     508, 516, 552, 554, 561,
                     567, 586, 591, 595, 602,
                     624, 628, 635, 641, 656, 
                     659, 666, 670, 674, 679,
                     704, 712, 717, 743, 745,
                     746, 758, 763, 772)

# 4x4 Squared (sq) patches
Mi1_sq_1_ii = c(758,406,567,344,
                129,759,673,403,
                616,411,551,45,
                52,615,438,407)

Mi1_sq_2_ii = c(62,149,609,398,
                70,78,231,518,
                592,90,473,32,
                213,542,227,206)

# Hexagonal (hx) patches
Mi1_hx_1_ii = c(129,344,229,320,759,615,52) #Dorsal in medulla
Mi1_home_1 = 129

Mi1_hx_2_ii = c(542,609,213,227,70,398,206) #Ventral in medulla
Mi1_home_2 = 542

# 1st 5x5 square around hexagons (1sq_hx)
Mi1_1sq_hx_1_ii = c(Mi1_hx_1_ii,567,406,
                    407,438,403,673,
                    329,635,586,201,
                    449,221,196,353)
# erased: 277 616 148 561
Mi1_1sq_hx_2_ii = c(Mi1_hx_2_ii,149,107,
                    285,574,661,358,
                    755,78,231,518,
                    592,473,32,62)
# erased: 90 106  13 572
# 2nd 7x7 square around hexagons (2sq_hx)
Mi1_2sq_hx_1_ii = c(Mi1_1sq_hx_1_ii,
                    115,284,758,45,551,411,
                    550,211,499,768,772,122,
                    57,54,380,194,7,17,
                    385,212,261,602,508,628)
Mi1_2sq_hx_2_ii = c(Mi1_1sq_hx_2_ii,
                    569,198,195,587,158,39,
                    457,120,10,22,589,452,
                    175,654,723,399,394,127,
                    620,288,652,672,636,84)
# 3rd 9x9 square around hexagons (3sq_hx)
Mi1_3sq_hx_1_ii = c(Mi1_2sq_hx_1_ii,697,318,373,
                    77,220,641,624,416,218,223,
                    249,86,470,769,529,347,323,
                    309,454,579,325,205,556,660,
                    190,189,606,497,405,630,50,547)
Mi1_3sq_hx_2_ii = c(Mi1_2sq_hx_2_ii,582,623,668,
                    238,642,531,64,559,273,124,
                    418,580,506,31,734,691,761,
                    139,740,764,424,430,38,305,
                    328,372,37,339,112,121,619)



nopen3d()
spheres3d(Mi1_ar[,,"m10"], col = "gray",radius = 1000, lit=F)
spheres3d(Mi1_ar[Mi1_home_1,,"m10"], col = "blue",radius = 1100, lit=F)
spheres3d(Mi1_ar[Mi1_home_2,,"m10"], col = "blue",radius = 1100, lit=F)
spheres3d(Mi1_ar[Mi1_hx_1_ii,,"m10"], col = "green",radius = 1100, lit=F)
spheres3d(Mi1_ar[Mi1_hx_2_ii,,"m10"], col = "green",radius = 1100, lit=F)
spheres3d(Mi1_ar[Mi1_1sq_hx_1_ii,,"m10"], col = "darkorange",radius = 1100, lit=F)
spheres3d(Mi1_ar[Mi1_1sq_hx_2_ii,,"m10"], col = "darkorange",radius = 1100, lit=F)
spheres3d(Mi1_ar[Mi1_2sq_hx_1_ii,,"m10"], col = "darkred",radius = 1100, lit=F)
spheres3d(Mi1_ar[Mi1_2sq_hx_2_ii,,"m10"], col = "darkred",radius = 1100, lit=F)
spheres3d(Mi1_ar[Mi1_3sq_hx_1_ii,,"m10"], col = "darkblue",radius = 1100, lit=F)
spheres3d(Mi1_ar[Mi1_3sq_hx_2_ii,,"m10"], col = "darkblue",radius = 1100, lit=F)
rgl.snapshot('Mi1_selection.png', fmt = 'png')

# you can pick Mi1 columns with identify3d(Mi1_ar[,,"m10"]

## Interacting with columns
# spheres3d(Mi1_ar[,,"m10"], col = "green3", radius = 1000, lit = T)
# identify3d(Mi1_ar[,,"m10"])


## Work on syn for your set ii 
ii = Mi1_hx_1_ii# previous Mi1_center_ii, Mi1_hx_1_ii
outer_ii = Mi1_1sq_hx_1_ii# previous Mi1_st_outer_ii, Mi1_1sq_hx_1_ii

Mi1_set_segid = flywire_latestid(c(rownames(Mi1_ar[ii,,"m10"])), method = "leaves") # "matrix" "array"
Mi1_set_syn =flywire_partners(Mi1_set_segid, partners = "both", details = T)# "data.frame" all synapses, inputs and outputs
Mi1_set_syn_pre = Mi1_set_syn %>% filter(prepost == 0 & pre_id != post_id)# "data.frame"  only pre synapses, outputs, no autapses

## Visualize cleft_score from Buhmann:
hist(Mi1_set_syn_pre$cleft_scores, breaks = 50)

# focusing on the synapses on the Mi1 tip
split_syn_data_t4_set <- split(Mi1_set_syn_pre,f=Mi1_set_syn_pre$query)
t4_set = data.frame()
M10_set = data.frame()
xyz = data.frame()

for (i in 1:length(split_syn_data_t4_set)) {
  xyz = data.frame(x = split_syn_data_t4_set[[names(split_syn_data_t4_set[i])]][["pre_x"]],
                   y = split_syn_data_t4_set[[names(split_syn_data_t4_set[i])]][["pre_y"]],
                   z = split_syn_data_t4_set[[names(split_syn_data_t4_set[i])]][["pre_z"]],
                   pre_id= split_syn_data_t4_set[[names(split_syn_data_t4_set[i])]][["pre_id"]],
                   post_id= split_syn_data_t4_set[[names(split_syn_data_t4_set[i])]][["post_id"]])
  
  N <- nrow(xyz) 
  mean_xyz <- apply(xyz[,1:3], 2, mean)
  xyz_pca   <- princomp(xyz[,1:3])
  dirVector <- xyz_pca$loadings[, 1]   # PC1
  xyz_fit <- matrix(rep(mean_xyz, each = N), ncol=3) + xyz_pca$score[, 1] %*% t(dirVector) 
  
  minx <- quantile(xyz_fit[,1], c(0.15))
  
  minindex <- which(xyz_fit[,1] < minx)
  
  # returns all segments with more then 2 synapses 
  min_xyz <- xyz[minindex,] %>%  group_by(post_id) %>% summarise(n = n()) %>% arrange(desc(n)) %>%  filter(n>1) # it was: filter(n>2)
  M10_min_xyz <- xyz[minindex,] %>%  group_by(post_id,pre_id) %>% summarise(n = n()) %>% arrange(desc(n)) %>%  filter(n>1) # it was: filter(n>2)
  
  # returns the top 6 partners  
  #min_xyz = xyz[minindex,] %>%  group_by(post_id) %>% summarise(n = n()) %>%  slice_max(n,n=6)
  t4_set <- rbind(t4_set,min_xyz)
  M10_set <- rbind(M10_set,M10_min_xyz )
}

t4_set$post_id = as.character(t4_set$post_id)
t4_set = t4_set %>% left_join(google_sheet, by = c("post_id"="seg_id"))

M10_set$post_id = as.character(M10_set$post_id)
M10_set = M10_set %>% left_join(google_sheet, by = c("post_id"="seg_id"))
M10_set_known = M10_set %>%  filter(n>0 & !is.na(type))
M10_set_known = M10_set_known %>%  filter(pre_id == '720575940637431110') # Insert the desired ID.

## Generating user-defined URL in flywire.
segment_id = as.character(M10_set_known$post_id) # the id list has to be character, very important
fw_url=with_segmentation('flywire', getOption('fafbseg.sampleurl'))
ngl_segments(fw_url)
fw_sc=ngl_decode_scene(fw_url)
fw_sc$layers[[2]]$segments= segment_id
browseURL(ngl_encode_url(fw_sc))

# partners that haven't been IDed yet
known = t4_set %>%  filter(n>0 & !is.na(type))
unknown = t4_set %>%  filter(n>0 & is.na(type)) # Seb: this needs to be added to google docs file
# flywire_url_func(unknown$post_id)

## remove duplications and arrange neurons according to type, hemisphere & name.
known  = known  %>% distinct(post_id, .keep_all = TRUE) %>% arrange(type,hemisphere,name)

# save known data set
sheet_write(known, ss="1QsdNNRLJA6vejNp1aM8n-T6WgFHE80PnWn_ptnFuGfA", sheet = "Sheet1")

# all Mi1 > T4 synapses
Mi1_set_syn_pre$post_id = as.character(Mi1_set_syn_pre$post_id)
Mi1_t4_syn = Mi1_set_syn_pre %>% 
  left_join(google_sheet, by = c("post_id"="seg_id")) %>% 
  filter(type=="T4") %>% 
  group_by(query,post_id) %>% 
  summarise(n = n()) %>% 
  arrange(query, desc(n))

# T4 meshes & synapses 
t4_set_msh = read_cloudvolume_meshes(unique(Mi1_t4_syn$post_id))
t4_set_syn =flywire_partners(unique(Mi1_t4_syn$post_id), partners = "both", details = T)
t4_set_syn_fl = t4_set_syn %>% filter(cleft_scores>50)

# restrict syn to neuropils, may fail for larger patches due to bad neuropil mshs
xxme= pointsinside(t4_set_syn_fl[,3:5],surf=subset(FAFB14NP.surf, "ME_L"))
xxlop= pointsinside(t4_set_syn_fl[,3:5],surf=subset(FAFB14NP.surf, "LOP_L"))

spheres3d(t4_set_syn_fl[t4_set_syn_fl$prepost==1,c(6:8)], col = "red",radius = 100, lit=F) # prepost = 1, postsynaptic, INPUTS
spheres3d(t4_set_syn_fl[t4_set_syn_fl$prepost==0,c(6:8)], col = "green",radius = 100, lit=F)# prepost = 0, pressynaptic, OUTPUTS
spheres3d(t4_set_syn_fl[xxme,c(6:8)], col= "grey",radius = 100, lit=F) # 
spheres3d(t4_set_syn_fl[xxlop,c(6:8)], col = "cyan",radius = 100, lit=F)

# T4 dentrite centroid 
t4_set_centroid = t4_set_syn_fl[xxme,] %>% filter(prepost==0) %>% group_by(query) %>% summarise(across(post_x:post_z, mean)) #prepost == 0, presynaptic, OUTPUTS
t4_set_centroid_in = t4_set_syn_fl[xxme,] %>% filter(prepost==1) %>% group_by(query) %>% summarise(across(post_x:post_z, mean)) #prepost == 1, postynaptic, INPUTS

spheres3d(t4_set_syn_fl[xxme,c(6:8)], col= "grey",radius = 100, lit=F)
spheres3d(t4_set_centroid[,2:4], col = "red",radius = 300, lit=F)
#spheres3d(t4_set_centroid_in[,2:4], col = "blue",radius = 300, lit=F)
spheres3d(Mi1_ar[ii,,"m10"], col = "darkgreen",radius = 500, lit=F)
spheres3d(Mi1_ar[outer_ii,,"m10"], col = "blue",radius = 500, lit=F)



# transform for plotting
pca_set = rbind(Mi1_ar[ii,,"m10"],Mi1_ar[ii,,"m0"]) 

mi1_center_pca = princomp(pca_set)
t4_set_syn <- sweep(as.matrix(t4_set_syn_fl[,6:8]), 2, mi1_center_pca$center) %*% mi1_center_pca$loadings %>% as.data.frame() # to each point (P): (P-mean)*eigenvector from pca
mi1_set_xform <- sweep(as.matrix(Mi1_ar[ii,,"m10"]), 2, mi1_center_pca$center) %*% mi1_center_pca$loadings %>% as.data.frame()
mi1_outer_xform <- sweep(as.matrix(Mi1_ar[outer_ii,,"m10"]), 2, mi1_center_pca$center) %*% mi1_center_pca$loadings %>% as.data.frame()
t4_set_centroid_xform <- sweep(as.matrix(t4_set_centroid[,2:4]), 2, mi1_center_pca$center) %*% mi1_center_pca$loadings %>% as.data.frame()

# estimate of T4 per column 
mi1 = rbind(mi1_set_xform,mi1_outer_xform) %>% 
  select(Comp.2,Comp.3) %>% 
  rename(x_mi1 = Comp.2, y_mi1 = Comp.3) %>% 
  mutate(y_mi1 = -y_mi1)
mi1$type = "mi1_center"

t4 = t4_set_centroid_xform  %>% 
  select(Comp.2,Comp.3) %>% 
  rename(x_t4 = Comp.2, y_t4 = Comp.3) %>% 
  mutate(y_t4 = -y_t4)
t4$type = "t4_center"

ff = crossing(t4,mi1,.name_repair = "unique") %>% 
  mutate(dist = sqrt((x_t4-x_mi1)^2 + (y_t4-y_mi1)^2)) %>%
  group_by(x_t4) %>% 
  slice_min(dist) %>% 
  group_by(x_mi1) %>% 
  mutate(n = n())

# plot data
ggplot(t4_set_syn[xxme,],aes(x = Comp.2, y = -Comp.3)) + 
  geom_point(size=0.1, col="grey") +
  geom_point(data = mi1_set_xform, size=2, col="darkgreen") +
  geom_point(data = mi1_outer_xform, size=2, col="darkblue") +
  geom_point(data = t4_set_centroid_xform, size=1, col="darkred") +
  geom_text(data = ff, aes(x = x_mi1, y = y_mi1, label = n), size=1.5, col="white") +
  coord_fixed() +
  theme_void()


ggplot(t4_set_syn[xxme,],aes(x = Comp.2, y = -Comp.3))  +
  stat_density_2d(geom = "raster", aes(fill = stat(density)), contour = FALSE) +
  scale_fill_gradient(low = "white", high = "#ed217c") +
  coord_fixed() +
  theme_void() +
  theme(legend.position = "none") 

#------------------------------------------------------------------------------------------------------
## Seb working with raw z
minz <- quantile(xyz$z, c(0.85))
minindex_z <- which(xyz$z > minz)
noncommon_indexes = c(setdiff(minindex,minindex_z), setdiff(minindex_z,minindex))

# returns all segments with more then 2 synapses
min_xyz_1 <- xyz[minindex,]
min_xyz_2 <- xyz[minindex_z,]
min_xyz_3 <- xyz[noncommon_indexes,]

## Seb visualization of the 15% quantile 
#--- SCORES

plot(xyz_pca$scores[,][,1],xyz_pca$scores[,][,2],xlim = c(- 20000, 40000),ylim = c(- 10000, 10000))
min_pca1_scores_ii = which(xyz_pca$scores[,][,1] < quantile(xyz_pca$scores[,][,1], c(0.15)))
min_pca2_scores_ii = which(xyz_pca$scores[,][,2] < quantile(xyz_pca$scores[,][,2], c(0.15)))
min_pca1_scores = xyz_pca$scores[,][,1][min_pca1_scores_ii]
min_pca2_scores = xyz_pca$scores[,][,2][min_pca2_scores_ii]
plot(min_pca1_scores,min_pca2_scores,xlim = c(- 20000, 40000),ylim = c(- 10000, 10000))

#--- XYZ_FIT
min_xyz_fit_ii = which(xyz_fit[,1] < quantile(xyz_fit[,1], c(0.90)))
min_xyz_fit = xyz_fit[min_xyz_fit_ii,]
plot_ly(x = xyz_fit[,1],y= xyz_fit[,2],z = xyz_fit[,3],type="scatter3d")
plot_ly(x = min_xyz_fit[,1],y= min_xyz_fit[,2],z = min_xyz_fit[,3],type="scatter3d")

xyz_fit_df <- as.data.frame(xyz_fit)
min_xyz_fit_df <- as.data.frame(min_xyz_fit)
plot_ly(xyz_fit_df, x = ~x, y = ~y,color = ~z, type = "scatter")

xyz_fit_df$gr <- 1
min_xyz_fit_df$gr <- 2
df <- rbind(xyz_fit_df, min_xyz_fit_df)
plot_ly(df, x = ~x, y = ~y,color = ~factor(gr), type = "scatter")

#--- RAW XYZ  
xyz$gr <- 1
min_xyz_1$gr <- 2
min_xyz_2$gr <- 3
min_xyz_3$gr <- 4
df <- rbind(xyz,min_xyz_1,min_xyz_2,min_xyz_3)
plot_ly(df, x = ~x, y = ~z,color = ~factor(gr), type = "scatter", alpha = 0.5)
# ----------------------------------------------------------------------------------------------------------

## Seb identifying single T4 centroids
# centroid_ls <- identify3d(t4_set_centroid[,2:4])
# t4_set_centroid_in[centroid_ls,]$query


# ----------------------------------------------------------------------------------------------------------

## Seb, some plotting of specific neurons
# Do it for every unique Mi1 from the M10_set
Mi1_home_1_id = 720575940613655775 
