Docs: [GoDoc](https://pkg.go.dev/github.com/emer/emergent/elog)

`elog` provides a full infrastructure for recording data of all sorts at multiple time scales and evaluation modes (training, testing, validation, etc).

The `elog.Item` provides a full definition of each distinct item that is logged with a map of compute functions keyed by a scope string that reflects the time scale and mode.  The same function can be used across multiple scopes, or a different function for each scope, etc.

The Items are then processed in `CreateTables()` to create a set of `etable.Table` tables to hold the data.

The `elog.Logs` struct holds all the relevant data and functions for managing the logging process.

* `Log(mode, time)` does logging, adding a new row

* `LogRow(mode, time, row)` does logging at given row

Both of these functions automatically write incrementally to a `tsv` File if it has been opened.

The `Context` object is passed to the Item Compute functions, and has all the info typically needed -- must call `SetContext(stats, net)` on the Logs to provide those elements.  Compute functions can do most standard things by calling methods on Context -- see that in Docs above for more info.


