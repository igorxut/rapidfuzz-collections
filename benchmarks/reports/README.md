# Benchmark Reports

This directory holds JSON/CSV output written by the benchmark scripts documented in [`../README.md`](../README.md).

Each report subdirectory corresponds to one benchmark run or one focused investigation matrix and normally contains:

- `<stem>.json`: result rows plus environment metadata;
- `<stem>.csv`: result rows only.

Report data is generated locally.  
Regenerate a report by running the corresponding script with the flags shown in [`../README.md`](../README.md).  
Historical figures quoted in [`../DESIGN.md`](../DESIGN.md) describe specific past local runs, not a committed artifact you can diff against — re-run the same flags to check current behavior on your own machine.
