/*
 * Copyright 2026 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <cstdint>
#include <string>

#include "gtest/gtest.h"
#include "absl/flags/flag.h"
#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "xla/tsl/platform/env.h"

ABSL_FLAG(std::string, test_remote_dir,
          "",
          "Remote directory to write test files to.");

namespace torch_tpu {

absl::Status AtomicWriteToCacheFile(const std::string& cache_entry_path,
                                    const std::string& serialized_data);
absl::Status EnsureDirExistsRecursively(const std::string& path);

namespace {

class AtomicWriteToCacheFileTest : public testing::Test {
 protected:
  ~AtomicWriteToCacheFileTest() override {
    // Clean up the test file if it exists.
    if (!test_file_.empty()) {
      env_->DeleteFile(test_file_).IgnoreError();
    }
    // Clean up the test directory if it exists.
    if (!test_dir_.empty()) {
      int64_t undeleted_files, undeleted_dirs;
      env_->DeleteRecursively(test_dir_, &undeleted_files, &undeleted_dirs)
          .IgnoreError();
    }
  }

  tsl::Env* const env_ = tsl::Env::Default();
  std::string test_file_;
  std::string test_dir_;
};

// Tests that AtomicWriteToCacheFile correctly writes a new file.
TEST_F(AtomicWriteToCacheFileTest, WriteNewFile) {
  test_file_ = absl::StrCat(absl::GetFlag(FLAGS_test_remote_dir), "/new_file.");
  // Make test_file_ unique in case the test is run with --runs_per_test=N.
  ASSERT_TRUE(env_->CreateUniqueFileName(&test_file_, /*suffix=*/".data"));
  const std::string content = "test_data_content";

  // Clean up the file if it exists.
  env_->DeleteFile(test_file_).IgnoreError();

  // Write the file.
  ABSL_LOG(INFO) << "Writing to " << test_file_;
  absl::Status status = AtomicWriteToCacheFile(test_file_, content);
  ASSERT_EQ(status, absl::OkStatus());
  ASSERT_EQ(env_->FileExists(test_file_), absl::OkStatus());

  // Verify content.
  std::string read_content;
  status = tsl::ReadFileToString(env_, test_file_, &read_content);
  ASSERT_EQ(status, absl::OkStatus());
  EXPECT_EQ(read_content, content)
      << "Unexpected content in file " << test_file_;
}

// Tests that AtomicWriteToCacheFile correctly overwrites an existing file.
TEST_F(AtomicWriteToCacheFileTest, OverwriteExistingFile) {
  test_file_ =
      absl::StrCat(absl::GetFlag(FLAGS_test_remote_dir), "/existing_file");
  // Make test_file_ unique in case the test is run with --runs_per_test=N.
  ASSERT_TRUE(env_->CreateUniqueFileName(&test_file_, /*suffix=*/".data"));
  const std::string content = "test_data_content";

  // Clean up the file if it exists.
  env_->DeleteFile(test_file_).IgnoreError();

  // Write the file.
  ABSL_LOG(INFO) << "Writing to " << test_file_;
  absl::Status status = AtomicWriteToCacheFile(test_file_, content);
  ASSERT_EQ(status, absl::OkStatus());
  ASSERT_EQ(env_->FileExists(test_file_), absl::OkStatus());

  // Overwrite the file.
  ABSL_LOG(INFO) << "Overwriting " << test_file_;
  const std::string new_content = "new_test_data_content";
  status = AtomicWriteToCacheFile(test_file_, new_content);
  ASSERT_EQ(status, absl::OkStatus());
  ASSERT_EQ(env_->FileExists(test_file_), absl::OkStatus());

  // Verify new content.
  std::string read_content;
  status = tsl::ReadFileToString(env_, test_file_, &read_content);
  ASSERT_EQ(status, absl::OkStatus());
  EXPECT_EQ(read_content, new_content)
      << "Unexpected content in file " << test_file_;
}

TEST_F(AtomicWriteToCacheFileTest, CreateDirectory) {
  test_dir_ = absl::StrCat(absl::GetFlag(FLAGS_test_remote_dir), "/new_dir");
  // Make test_dir_ unique in case the test is run with --runs_per_test=N.
  ASSERT_TRUE(env_->CreateUniqueFileName(&test_dir_, /*suffix=*/""));
  // Add two more levels of directories to test recursive creation.
  const std::string test_subdir = absl::StrCat(test_dir_, "/level1/level2");

  // Clean up test_dir_ if it exists.
  int64_t undeleted_files, undeleted_dirs;
  env_->DeleteRecursively(test_dir_, &undeleted_files, &undeleted_dirs)
      .IgnoreError();

  // Create the directory.
  ABSL_LOG(INFO) << "Creating directory " << test_subdir;
  absl::Status status = EnsureDirExistsRecursively(test_subdir);

  // Verify the directory was created.
  ASSERT_EQ(status, absl::OkStatus());
  ASSERT_EQ(env_->IsDirectory(test_subdir), absl::OkStatus());
}

}  // namespace
}  // namespace torch_tpu
