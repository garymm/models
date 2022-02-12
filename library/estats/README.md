Docs: [GoDoc](https://pkg.go.dev/github.com/emer/emergent/estats)

`estats.Stats` provides maps for storing statistics as named scalar and tensor values.  These stats are available in the elog.Context for use during logging.

A common use-case for example is to use `F32Tensor` to manage a tensor that is reused every time you need to access values on a given layer:

```Go
    ly := ctxt.Net.LayerByName(lnm)
    tsr := ctxt.Stats.F32TEnsorr(lnm)
    ly.UnitValsTensor(tsr, "Act")
    // tsr now has the "Act" values from given layer -- can be logged, computed on, etc..
```

