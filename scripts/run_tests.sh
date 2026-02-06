#!/bin/bash
#
# Fano Test Runner
#
# Runs all tests with proper isolation for module conflicts.
# Pool tests and LLM tests must be run separately due to
# conflicting src/models.py modules.
#
# Usage:
#   ./scripts/run_tests.sh           # Run all tests
#   ./scripts/run_tests.sh --quick   # Run only shared/control tests
#   ./scripts/run_tests.sh --cov     # Run with coverage report
#

set -e

cd "$(dirname "$0")/.."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

QUICK_MODE=false
COVERAGE_MODE=false

# Parse arguments
for arg in "$@"; do
    case $arg in
        --quick)
            QUICK_MODE=true
            ;;
        --cov|--coverage)
            COVERAGE_MODE=true
            ;;
    esac
done

echo -e "${YELLOW}=== Fano Test Runner ===${NC}"
echo ""

# Coverage flags
COV_FLAGS=""
if [ "$COVERAGE_MODE" = true ]; then
    COV_FLAGS="--cov=shared --cov=control --cov=explorer --cov=documenter --cov-report=term-missing"
    echo -e "${YELLOW}Coverage mode enabled${NC}"
fi

run_test_suite() {
    local name=$1
    local path=$2
    local flags=${3:-""}

    echo ""
    echo -e "${YELLOW}=== Running $name Tests ===${NC}"

    if pytest "$path" -v --tb=short $flags $COV_FLAGS; then
        echo -e "${GREEN}$name tests passed${NC}"
        return 0
    else
        echo -e "${RED}$name tests FAILED${NC}"
        return 1
    fi
}

FAILED=0

# Shared utilities (foundational - run first)
run_test_suite "Shared" "tests/shared/" || FAILED=$((FAILED + 1))

# Control module
run_test_suite "Control" "tests/control/" || FAILED=$((FAILED + 1))

if [ "$QUICK_MODE" = true ]; then
    echo ""
    echo -e "${YELLOW}Quick mode: skipping remaining test suites${NC}"
else
    # Documenter tests
    if [ -d "tests/documenter" ] && ls tests/documenter/test_*.py 1> /dev/null 2>&1; then
        run_test_suite "Documenter" "tests/documenter/" || FAILED=$((FAILED + 1))
    fi

    # Explorer tests
    if [ -d "tests/explorer" ] && ls tests/explorer/test_*.py 1> /dev/null 2>&1; then
        run_test_suite "Explorer" "tests/explorer/" || FAILED=$((FAILED + 1))
    fi

    # Researcher tests
    if [ -d "tests/researcher" ] && ls tests/researcher/test_*.py 1> /dev/null 2>&1; then
        run_test_suite "Researcher" "tests/researcher/" || FAILED=$((FAILED + 1))
    fi

    # API tests
    if [ -d "tests/api" ] && ls tests/api/*/test_*.py 1> /dev/null 2>&1; then
        run_test_suite "API" "tests/api/" || FAILED=$((FAILED + 1))
    fi

    # E2E tests (separate - may be slow)
    if [ -d "tests/e2e" ] && ls tests/e2e/test_*.py 1> /dev/null 2>&1; then
        run_test_suite "E2E" "tests/e2e/" "-m e2e" || FAILED=$((FAILED + 1))
    fi

    # Orchestrator tests (separate from pool/llm)
    if [ -d "tests/orchestrator" ] && ls tests/orchestrator/test_*.py 1> /dev/null 2>&1; then
        run_test_suite "Orchestrator" "tests/orchestrator/" || FAILED=$((FAILED + 1))
    fi

    # Pool tests (ISOLATED - module conflict with llm)
    echo ""
    echo -e "${YELLOW}=== Running Pool Tests (isolated) ===${NC}"
    echo -e "Note: Pool tests run separately due to module naming conflict"
    if pytest pool/tests/ -v --tb=short; then
        echo -e "${GREEN}Pool tests passed${NC}"
    else
        echo -e "${RED}Pool tests FAILED${NC}"
        FAILED=$((FAILED + 1))
    fi

    # LLM tests (ISOLATED - module conflict with pool)
    echo ""
    echo -e "${YELLOW}=== Running LLM Tests (isolated) ===${NC}"
    echo -e "Note: LLM tests run separately due to module naming conflict"
    if pytest llm/tests/ -v --tb=short; then
        echo -e "${GREEN}LLM tests passed${NC}"
    else
        echo -e "${RED}LLM tests FAILED${NC}"
        FAILED=$((FAILED + 1))
    fi
fi

# Summary
echo ""
echo -e "${YELLOW}=== Test Summary ===${NC}"

if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN}All test suites passed!${NC}"
    exit 0
else
    echo -e "${RED}$FAILED test suite(s) failed${NC}"
    exit 1
fi
