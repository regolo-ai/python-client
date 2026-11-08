# Test fixtures

`devmid-3.9-public-openapi.json` describes the Devmid 3.9 public user API. It is
used by management tests, schema discovery tests, documentation checks and
`scripts/check_devmid_contract.py`. The client does not load it at runtime.

When updating this fixture, include only public user operations and their
request/response schemas. Use test data; exclude production records, credentials
and private server configuration. See [contributing](../../docs/development.md)
for contract checks and update instructions.
