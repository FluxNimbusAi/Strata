// pool_quota_affinity_test: the CPU count the pool plans when a container quota meets an affinity mask
// (include/strata/kernels/cpu/cgroup_cpu.hpp + pool_plan.hpp), plus the list shape PR #14 restored.
//
// What upstream's suites already cover, and the gap this closes:
//   * cgroup_cpu_test.cpp  - the quota loader alone ("400000 100000" -> 4.0 CPUs), no worker plan.
//   * pool_plan_test.cpp   - the quota clamp alone (a raw PoolPlanInput.cpu_quota), never a measured fixture.
//   * the affinity suite   - mask round-trips and membership, but it never asserts the LENGTH of the core list.
// Here the two halves run chained - a quota measured from an on-disk cgroup fixture feeds the plan built from an
// affinity-masked core list - and the mask -> core-list projection is checked against the dangling-else shape of
// issue #13 (a restricted mask of k bits produced k + (1024 - k) * h entries; the reported box showed 16143).
//
// Pure logic plus a temp directory - no threads, no real cgroups, no /sys - so it runs anywhere, as cgroup_cpu_test.
// Which assertions fail on the pre-fix behavior:
//   * the quota clamp in plan_pool_workers (5ece068f7 introduced pool_plan.hpp with it; the plan before it knew
//    only "every physical core but the host's"): with the clamp taken back out, every quota case below returns the
//     unclamped affinity count - 15 where 3 is expected, 15 where 1, 15 where 2 - issue #11's oversubscribed
//     spinning pool exactly, and the limit string names the cores instead of "the container's CPU quota".
//   * PR #14: `cores_pre_pr14` reproduces the loop of 6da1f667e verbatim; the mask -> list cases below then blow
//     the list out to k + (1024 - k) * h - the 16143 of issue #13 - so any fixed-shape assertion is red there.
#include "strata/kernels/cpu/cgroup_cpu.hpp"
#include "strata/kernels/cpu/pool_plan.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace cpu = strata::kernels::cpu;
namespace fs = std::filesystem;

static int g_fail = 0;

static void expect(bool ok, const std::string& what) {
    if (!ok) {
        ++g_fail;
        std::fprintf(stderr, "FAIL: %s\n", what.c_str());
    }
}

static bool near(double a, double b) { return std::fabs(a - b) < 1e-9; }

static void put(const fs::path& file, const std::string& text) {
    fs::create_directories(file.parent_path());
    std::ofstream(file) << text;
}

// the mask projection's unit size: CPU_SETSIZE on Linux (the producer's bound), 1024 as the fallback the
// historical loop used - CPU_SETSIZE is 1024 on glibc, which is what made issue #13's count reproducible.
static const int kMaskBits = 1024;

// The mask -> core-list projection AFTER PR #14: every bit of the affinity mask becomes one entry and no
// second loop runs over the unset bits.  `cpus` is the mask's bit numbers in ascending order, `hw` what
// std::thread::hardware_concurrency() reports for the fallback loop.
static std::vector<int> cores_projected(const std::vector<int>& cpus, int hw) {
    std::vector<int> cores;
    for (int cpu_id : cpus) cores.push_back(cpu_id);
    if (cores.empty())
        for (int i = 0; i < hw; ++i) cores.push_back(i);   // the else branch of the sched_getaffinity check
    return cores;
}

// The projection as it was BEFORE PR #14 (commit 6da1f667e, src/kernels/cpu/pool.cpp) - the dangling else let
// the fallback loop run once per UNSET bit of the fixed 1024-entry mask.  Verbatim shape:
//     if (sched_getaffinity(0, sizeof set, &set) == 0)
//         for (int i = 0; i < CPU_SETSIZE; ++i)
//             if (CPU_ISSET(i, &set)) cores.push_back(i);
//     else
//         for (unsigned i = 0; i < hardware_concurrency(); ++i) cores.push_back(i);
static std::vector<int> cores_pre_pr14(const std::vector<int>& cpus, int hw) {
    std::vector<int> cores;
    for (int i = 0; i < kMaskBits; ++i) {
        if (std::find(cpus.begin(), cpus.end(), i) != cpus.end()) {
            cores.push_back(i);
        } else {
            for (int h = 0; h < hw; ++h) cores.push_back(h);
        }
    }
    return cores;
}

// The plan for a process allowed on `affinity_cores` physical cores (one package, the host's core counted in -
// the mask is the process's whole set) under a container quota.  pool_quota() also returns what bounded it.
static cpu::PoolPlan plan_quota(int affinity_cores, double quota) {
    cpu::PoolPlanInput in;
    in.logical = affinity_cores;
    in.cores_per_package = {affinity_cores};
    in.cpu_quota = quota;
    return cpu::plan_pool_workers(in);
}

static int plan_workers(int affinity_cores, double quota) { return plan_quota(affinity_cores, quota).workers; }

int main() {
    const fs::path root = fs::temp_directory_path() /
        ("strata_pool_quota_affinity_test_" + std::to_string((long long) std::time(nullptr)));
    fs::remove_all(root);
    const std::string r = root.string();
    std::printf("pool_quota_affinity_test: cap %d, reserve %d from %d\n",
                cpu::kPoolWorkersCap, cpu::kPoolReserve, cpu::kPoolReserveFrom);

    // ------------------------------------------------------------------ the quota fixtures, measured from disk
    // a container with its own namespace: cpu.max at the mount's root, 4 CPUs
    put(root / "q4" / "cpu.max", "400000 100000\n");
    expect(near(cpu::cgroup_cpu_quota_at((root / "q4").string(), "0::/\n"), 4.0), "fixture: quota 4 CPUs");
    // quota 12 CPUs (larger than the affinity below)
    put(root / "q12" / "cpu.max", "1200000 100000\n");
    expect(near(cpu::cgroup_cpu_quota_at((root / "q12").string(), "0::/\n"), 12.0), "fixture: quota 12 CPUs");
    // no limit at all, both ways: cpu.max says "max", and the file is simply not there
    put(root / "qmax" / "cpu.max", "max 100000\n");
    expect(near(cpu::cgroup_cpu_quota_at((root / "qmax").string(), "0::/\n"), 0.0), "fixture: 'max' -> no quota");
    expect(near(cpu::cgroup_cpu_quota_at((root / "qnone").string(), "0::/\n"), 0.0), "fixture: no cpu.max -> no quota");
    // a sub-CPU quota, and a huge one
    put(root / "qh" / "cpu.max", "50000 100000\n");
    expect(near(cpu::cgroup_cpu_quota_at((root / "qh").string(), "0::/\n"), 0.5), "fixture: quota 0.5 CPUs");
    put(root / "q200" / "cpu.max", "20000000 100000\n");
    expect(near(cpu::cgroup_cpu_quota_at((root / "q200").string(), "0::/\n"), 200.0), "fixture: quota 200 CPUs");
    // the quota on a PARENT of the process's group (a systemd CPUQuota=): the mount's root and the scope say
    // "max", system.slice carries 4 CPUs, and the tightest limit on the way up wins
    put(root / "host" / "cpu.max", "max 100000\n");
    put(root / "host" / "system.slice" / "cpu.max", "400000 100000\n");
    put(root / "host" / "system.slice" / "docker-abc.scope" / "cpu.max", "max 100000\n");
    const double parent_quota =
        cpu::cgroup_cpu_quota_at((root / "host").string(), "0::/system.slice/docker-abc.scope\n");
    expect(near(parent_quota, 4.0), "fixture: the parent's 4 CPUs");
    // cgroup v1: the group itself says -1 (no limit), the parent docker/ carries 3 CPUs
    put(root / "v1" / "cpu,cpuacct" / "cpu.cfs_quota_us", "-1\n");
    put(root / "v1" / "cpu,cpuacct" / "cpu.cfs_period_us", "100000\n");
    put(root / "v1" / "cpu,cpuacct" / "docker" / "cpu.cfs_quota_us", "300000\n");
    put(root / "v1" / "cpu,cpuacct" / "docker" / "cpu.cfs_period_us", "100000\n");
    put(root / "v1" / "cpu,cpuacct" / "docker" / "abc" / "cpu.cfs_quota_us", "-1\n");
    put(root / "v1" / "cpu,cpuacct" / "docker" / "abc" / "cpu.cfs_period_us", "100000\n");
    const double v1_quota = cpu::cgroup_cpu_quota_at(
        (root / "v1").string(), "12:memory:/docker/abc\n4:cpu,cpuacct:/docker/abc\n");
    expect(near(v1_quota, 3.0), "fixture: v1, the parent's 3 CPUs");

    // ------------------------------------------------------------------ quota x affinity, chained through the plan
    // quota 4 CPUs under an affinity mask of 16 cores: THE REGRESSION (issue #11).  Without the clamp the plan
    // ignored the quota and returned 15 - the workers that spin past --cpus 4 and throttle each other.
    const double q4 = cpu::cgroup_cpu_quota_at((root / "q4").string(), "0::/\n");
    const cpu::PoolPlan p4 = plan_quota(16, q4);
    expect(p4.workers == 3, "quota 4 CPUs with a 16-core affinity mask -> 3 workers");
    expect(std::string(p4.limit) == "the container's CPU quota", "the 4-CPU quota names itself as the limit");
    expect(q4 > 0.0 && plan_workers(16, q4) < plan_workers(16, 0.0), "the quota binds BELOW the affinity count");
    // quota absent: every physical core but the host's (16 cores -> 15) - the old behavior, unchanged
    expect(plan_workers(16, cpu::cgroup_cpu_quota_at((root / "qnone").string(), "0::/\n")) == 15,
           "no quota -> 15 workers on 16 cores");
    expect(plan_workers(16, cpu::cgroup_cpu_quota_at((root / "qmax").string(), "0::/\n")) == 15,
           "'max 100000' -> no quota -> 15 workers on 16 cores");
    // quota 12 CPUs LARGER than the affinity mask (8 cores): the mask bounds first and the quota is a no-op
    const double q12 = cpu::cgroup_cpu_quota_at((root / "q12").string(), "0::/\n");
    const cpu::PoolPlan p8q12 = plan_quota(8, q12);
    expect(p8q12.workers == 7, "quota 12 CPUs larger than an 8-core affinity mask -> 7 workers, the mask binds");
    expect(std::string(p8q12.limit) != "the container's CPU quota", "the no-op quota does not claim the limit");
    // the boundary: quota == the affinity count is still a no-op (16 cores: n = 16 - 1 = 15, q = 16 - 1 = 15,
    // and the clamp fires only strictly below n)
    put(root / "q16" / "cpu.max", "1600000 100000\n");
    expect(plan_workers(16, cpu::cgroup_cpu_quota_at((root / "q16").string(), "0::/\n")) == 15,
           "quota equal to the affinity count -> the plan is unchanged");
    // a sub-CPU quota floors at one worker, never zero
    expect(plan_workers(16, cpu::cgroup_cpu_quota_at((root / "qh").string(), "0::/\n")) == 1,
           "quota 0.5 CPUs -> 1 worker (never 0)");
    // a huge quota does not lift the measured cap (64 cores under a 200-CPU quota -> 32, 'the measured cap')
    {
        cpu::PoolPlanInput in;
        in.logical = 128;
        in.cores_per_package = {64};
        in.cpu_quota = cpu::cgroup_cpu_quota_at((root / "q200").string(), "0::/\n");
        const cpu::PoolPlan p = cpu::plan_pool_workers(in);
        expect(p.workers == cpu::kPoolWorkersCap, "quota 200 CPUs on 64 cores -> the cap, not the quota");
        expect(std::string(p.limit) == "the measured cap", "quota 200 CPUs: the limit names the cap");
    }
    // the quota on a PARENT cgroup clamps through the walk (4 CPUs on system.slice, the scope says "max")
    expect(plan_workers(16, parent_quota) == 3, "a parent scope's quota clamps: 4 CPUs -> 3 workers");
    // and the v1 loader's number clamps the same way (3 CPUs -> q = 2)
    expect(plan_workers(16, v1_quota) == 2, "v1 parent quota 3 CPUs -> 2 workers");
    // a quota that is present never changes the projected core LIST - it only bounds the count through the plan
    // (physical_cores() is affinity + topology only; the clamp lives in plan_pool_workers)
    {
        std::vector<int> mask;
        for (int i = 0; i < 16; ++i) mask.push_back(i);
        const std::vector<int> plain = cores_projected(mask, 16);
        expect(plain.size() == 16 && std::all_of(plain.begin(), plain.end(), [](int c) { return c < 16; }),
               "the projected list with a quota present is the mask itself: 16 entries, all allowed");
    }

    // ------------------------------------------------------------------ PR #14: the dangling-else shape, guarded
    // the mask -> core-list invariants the pool depends on (the bug multiplied the list per unset mask bit)
    const std::vector<int> full16 = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
    {
        const std::vector<int> l = cores_projected(full16, 16);
        expect((int) l.size() == 16, "a full 16-CPU mask projects to exactly 16 cores");
        expect(l == full16, "the projection keeps every allowed CPU, in order, without extras");
    }
    {   // a sparse mask - the pre-fix loop fed the fallback per unset bit, thousands of entries
        const std::vector<int> sparse = {0, 100};
        const std::vector<int> l = cores_projected(sparse, 16);
        expect(l == sparse, "a sparse mask {0, 100} projects to exactly {0, 100}");
    }
    {   // an unset/failed affinity query falls back to hardware_concurrency, once - not once per bit
        const std::vector<int> none;
        const std::vector<int> l = cores_projected(none, 8);
        expect((int) l.size() == 8, "no allowed CPUs (the else branch) -> hardware_concurrency entries, once");
    }
    // the golden count of issue #13: k + (1024 - k) * h, and for the box that reported it, minus the host core
    {
        const int k = (int) full16.size(), h = 16;
        const std::vector<int> bad = cores_pre_pr14(full16, h);
        expect((int) bad.size() == k + (kMaskBits - k) * h,
               "pre-PR-#14 projection: k + (1024 - k) * h entries for a k-bit mask");
        expect((int) bad.size() == 16144 && bad.size() - 1 == 16143,
               "pre-PR-#14 with issue #13's numbers: 16143 workers (16144 minus the host's core)");
        expect(cores_projected(full16, h).size() != cores_pre_pr14(full16, h).size(),
               "the fixed and dangling-else projections differ in length: the assertion catches the regression");
    }
    // what each half of the bug, alone, does to the plan - and how current main bounds both
    {
        // the ballooned 16144-entry list WITHOUT a quota: the cap is the last line of defense (16142 -> 32)
        expect(plan_workers(16144, 0.0) == cpu::kPoolWorkersCap,
               "a ballooned 16144-entry list with no quota still caps at 32");
        // the ballooned list WITH the 4-CPU quota: the clamp fires before the cap, 3 workers - issue #11's
        // spinning pool is bounded even if a list bug ever returns
        expect(plan_workers(16144, q4) == 3, "a ballooned list under the 4-CPU quota is clamped to 3");
        // and on the fixed 16-entry list the quota clamps to 3, not the 15 issue #11 observed
        expect((int) cores_projected(full16, 16).size() - 1 == 15 && plan_workers(16, q4) == 3,
               "fixed list + 4-CPU quota -> 3 workers, not the 15 of issue #11");
    }

    fs::remove_all(root);
    if (g_fail) {
        std::fprintf(stderr, "pool_quota_affinity_test: %d failure(s)\n", g_fail);
        return 1;
    }
    std::printf("pool_quota_affinity_test: OK\n");
    return 0;
}
