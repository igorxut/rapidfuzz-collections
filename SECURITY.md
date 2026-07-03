# Security Policy

## Scope

`rapidfuzz-collections` is a pure Python library. It:

- makes no network connections;
- does not read or write files;
- stores no credentials or sensitive data;
- executes no external processes.

The library operates only on data provided by the caller. Its attack surface is limited to the values passed in at runtime.

## Dependency security

This library depends on [RapidFuzz](https://github.com/rapidfuzz/RapidFuzz). Security issues in RapidFuzz should be reported directly to the RapidFuzz project.

## Reporting a security issue

If you believe you have found a security issue in `rapidfuzz-collections` itself, please open a [GitHub Issue](https://github.com/igorxut/rapidfuzz-collections/issues).

There is no private disclosure process for this project.
