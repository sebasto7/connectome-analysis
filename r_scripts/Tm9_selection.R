library(natverse)
library(tidyverse)
library(fafbseg)
library(googlesheets4)
library(glue)
library(cowplot)
library(plotly)
library(dplyr)

#If not e
#natmanager::install(pkgs="fafbseg")

choose_segmentation(release = 'flywire31')


### "google_sheet" table --------------------------------------------------

## Read the "google_sheet" table
ss_str = "1fZwH0qbVGT6vTt0kq3Z1IpniVdfMU1XuC72nUeLkD5Y" # Tm9s
google_sheet = read_sheet(ss_str)





## Checking if cell identity has been submitted
myids=flywire_ids('Tm9')
myids= google_sheet$seg_id

tsnow=flywire_timestamp(timestamp = Sys.time())
niv2=flywire_cave_query('neuron_information_v2', timestamp=tsnow)
niv2.dedup=niv2 %>% 
  group_by(pt_root_id) %>% 
  summarise(user_ids=paste(user_id, collapse = ','), 
            tags=paste(tag, collapse = ",")) %>% 
  mutate(pt_root_id=as.character(pt_root_id))
iddf=data.frame(pt_root_id=flywire_ids(myids))
iddf$pt_root_id=flywire_updateids(iddf$pt_root_id, timestamp=tsnow)
iddf2=left_join(iddf, niv2.dedup, by='pt_root_id')

iddf2 %>% 
  count(!is.na(tags))

iddf2 %>% 
  filter(is.na(tags))