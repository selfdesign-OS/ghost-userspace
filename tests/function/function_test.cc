#include <stdio.h>

#include <atomic>
#include <memory>
#include <vector>

#include "lib/base.h"
#include "lib/ghost.h"

namespace ghost {
    namespace {

        struct ScopedTime {
        ScopedTime() { start = absl::Now(); }
        ~ScopedTime() {
            printf(" took %0.2f ms\n", absl::ToDoubleMilliseconds(absl::Now() - start));
        }
        absl::Time start;
        };

        void New_01(int num_tasks) {
            std::vector<std::unique_ptr<GhostThread>> tasks;

            tasks.reserve(num_tasks);
            for (int i = 0; i < num_tasks; i++) {
            tasks.emplace_back(
                new GhostThread(GhostThread::KernelScheduler::kGhost, [i] {
                    // task
                }));
            }

            for (auto& task : tasks) task->Join();
        }

    }
}

int main() {
    {
        printf("new-01\n");
        ghost::ScopredTime time;
        ghost::New_01(15);
    }
    
    return 0;
}