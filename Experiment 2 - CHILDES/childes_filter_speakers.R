# -----------------------------------------------------------
# File: childes_filter_speakers.R
# Purpose: Allow manual filtering of which speakers the child hears
#          before extracting utterances for parsing.
# -----------------------------------------------------------
library(dplyr)

# Helper function to find most recent file
get_most_recent_file <- function(directory, pattern) {
  files <- list.files(directory, pattern = pattern, full.names = TRUE)
  if (length(files) == 0) stop("No files found matching pattern: ", pattern)
  files[which.max(file.info(files)$mtime)]
}

# Create timestamp
timestamp <- format(Sys.time(), "%Y%m%d_%H%M%S")

# 1. Load the most recent master speaker list
cat("Loading most recent adult speakers file...\n")
input_file <- get_most_recent_file(
  "rdata/speakers",
  "childes_adult_english_speakers_.*\\.csv"
)
cat("Loading:", input_file, "\n")
df <- read.csv(input_file)
cat("Loaded master list:", nrow(df), "non-child English speakers.\n\n")

# 2. Extract and print ALL roles (so you know what exists)
all_roles <- sort(unique(df$role))
cat("==== ALL ROLES IN DATA (copy these into the filter list) ====\n")
for (r in all_roles) cat(r, "\n")
cat("=============================================================\n\n")

# 3. *** MANUAL FILTERING LIST ***
roles_to_keep <- c(
  "Adult",
  "Caretaker",
  "Father",
  "Friend",
  "Grandfather",
  "Grandmother",
  "Investigator",
  "Mother",
  "Narrator",
  "Playmate",
  "Relative",
  "Sibling",
  "Sister",
  "Brother",
  "Teacher",
  "Unidentified",
  "Visitor",
  "Teenager",
  "Participant",
  "Girl",
  "Male",
  "Student",
  "Environment",
  "Doctor",
  "Target_Adult"
)

# 4. Filter the speaker table based on your list
filtered_speakers <- df[df$role %in% roles_to_keep, ]
cat("Speakers retained after filtering:", nrow(filtered_speakers), "\n")

# Remove bilingual corpora
filtered_speakers <- filtered_speakers %>%
  filter(!grepl("Biling", corpus_name, ignore.case = TRUE))
cat("After removing bilingual corpora:", nrow(filtered_speakers), "speakers\n")

# 5. Save filtered speaker table with timestamp
dir.create("rdata/speakers", showWarnings = FALSE, recursive = TRUE)
outfile <- paste0("rdata/speakers/childes_filtered_speakers_", timestamp, ".csv")
write.csv(filtered_speakers, outfile, row.names = FALSE)
cat("Filtered speaker list saved to:", outfile, "\n")

cat("\nRole distribution in filtered data:\n")
print(table(filtered_speakers$role))
