# -----------------------------------------------------------
# File: childes_get_utterances.R
# Purpose: Extract all utterances from the speakers you chose
#          in childes_filter_speakers.R and aggregate into
#          full utterances ready for spaCy parsing.
# -----------------------------------------------------------
library(childesr)
library(dplyr)

# Helper function to find most recent file
get_most_recent_file <- function(directory, pattern) {
  files <- list.files(directory, pattern = pattern, full.names = TRUE)
  if (length(files) == 0) stop("No files found matching pattern: ", pattern)
  files[which.max(file.info(files)$mtime)]
}

# Create timestamp
timestamp <- format(Sys.time(), "%Y%m%d_%H%M%S")

cat("Loading most recent filtered speaker list...\n")
input_file <- get_most_recent_file(
  "rdata/speakers",
  "childes_filtered_speakers_.*\\.csv"
)
cat("Loading:", input_file, "\n")
speakers <- read.csv(input_file)
cat("Speakers to extract:", nrow(speakers), "\n\n")

# Get unique combinations of corpus and role
unique_corpus_role <- speakers %>%
  select(corpus_name, role) %>%
  distinct()

cat("Fetching utterances from", nrow(unique_corpus_role), "corpus-role combinations...\n\n")

# Create an empty list to store utterances
utterance_list <- list()

# Loop through each unique corpus-role combination
for (i in seq_len(nrow(unique_corpus_role))) {

  corpus_name <- unique_corpus_role$corpus_name[i]
  role_name   <- unique_corpus_role$role[i]

  cat("Getting utterances for role:", role_name,
      "in corpus:", corpus_name, "...\n")

  # Pull utterances for this corpus and role
  utts <- tryCatch({
    get_utterances(corpus = corpus_name, role = role_name)
  }, error = function(e) {
    cat("  Error:", e$message, "\n")
    return(NULL)
  })

  # Skip if no utterances returned
  if (is.null(utts) || nrow(utts) == 0) {
    cat("  No utterances found.\n")
    next
  }

  cat("  Found", nrow(utts), "tokens.\n")

  # Append to list
  utterance_list[[length(utterance_list) + 1]] <- utts
}

cat("\nFinished downloading utterances.\n")

# Combine all utterances into one tibble
all_tokens <- bind_rows(utterance_list)
cat("Total tokens collected:", nrow(all_tokens), "\n")

# -----------------------------------------------------------
# Aggregate tokens into full utterances
# -----------------------------------------------------------
cat("\nAggregating tokens into full utterances...\n")

full_utterances <- all_tokens %>%
  group_by(transcript_id, speaker_id, utterance_order) %>%
  summarise(
    full_utterance = paste(gloss, collapse = " "),
    speaker_code = first(speaker_code),
    speaker_name = first(speaker_name),
    speaker_role = first(speaker_role),
    target_child_name = first(target_child_name),
    target_child_age = first(target_child_age),
    target_child_sex = first(target_child_sex),
    corpus_name = first(corpus_name),
    utterance_type = first(type),
    num_tokens = first(num_tokens),
    .groups = "drop"
  ) %>%
  arrange(transcript_id, utterance_order)

cat("Total full utterances:", nrow(full_utterances), "\n")

# Save the full utterances with timestamp
dir.create("rdata/utterances", showWarnings = FALSE, recursive = TRUE)
outfile <- paste0("rdata/utterances/childes_full_utterances_", timestamp, ".csv")
write.csv(full_utterances, outfile, row.names = FALSE)
cat("Full utterances saved to:", outfile, "\n")

# Preview first few utterances
cat("\nFirst 10 utterances:\n")
print(head(full_utterances$full_utterance, 10))
