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

#include "torch_tpu/common/op_name_stack.h"

#include <optional>
#include <stack>

#include "absl/log/absl_check.h"
#include "torch_tpu/ops/op_names.h"

namespace torch_tpu {
namespace internal {

// ClangTidy is wrong: a static and global variable is the right choice here
// even though it's not trivially destructible. This is a thread-local
// variable, so it needs to be destructed when the thread ends to avoid memory
// leaks. Hence we shouldn't use NoDestructor<> here.
thread_local  // CPP_THREAD_LOCAL_OK=needed only in the dispatching thread.
    std::stack<OpName>
        OpNameStack::stack_;  // NOLINT

void OpNameStack::Push(OpName op_name) { stack_.push(op_name); }

void OpNameStack::Pop() {
  ABSL_CHECK(!stack_.empty());  // CRASH_OK
  stack_.pop();
}

OpName OpNameStack::Top() {
  ABSL_CHECK(!stack_.empty());  // CRASH_OK
  return stack_.top();
}

std::optional<OpName> OpNameStack::MaybeTop() {
  if (stack_.empty()) {
    return std::nullopt;
  }
  return stack_.top();
}

}  // namespace internal
}  // namespace torch_tpu
