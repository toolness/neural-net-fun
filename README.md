This is a simple interactive lab that lets you play
around with multi-layer neural nets.

Its core engine is inspired by Andrej Karpathy's
[micrograd][] engine, and I made it to solidify
my understanding of the concepts he taught in
his [lecture](https://www.youtube.com/watch?v=VMj-3S1tku0).

You can access the web version at [toolness.github.io/neural-net-fun/](https://toolness.github.io/neural-net-fun/).

You can also follow a tutorial I wrote for it at [Atul’s Neural Net Fun Tutorial](https://toolness.notion.site/neural-net-fun-tutorial).

[micrograd]: https://github.com/karpathy/micrograd

## Quick start

To run it, you'll need to [install Rust](https://www.rust-lang.org/tools/install)
and run:

```
cargo run --release
```

Once the window opens, you can press `H` for help.

## Web version

To build the web version, run:

```
sh build-wasm.sh
```

Then run a web server (e.g. `basic-http-server`, installable via `cargo`) in the
root of the `dist` directory and visit it.

You can deploy the web version with `npm run deploy`.

## Android version

This is super experimental. You'll need Docker.

```
sh build-android.sh
```

This will build the app, attempt to install it into
the local android device, and run it via `adb`.

For details on how to debug, see [this Macroquad article](https://macroquad.rs/articles/android/#debug-logs).

## Related resources

- See also [perceptron-fun], my interactive lab for
  Rosenblatt's perceptron.

[perceptron-fun]: https://github.com/toolness/perceptron-fun

## License

Everything in this repository is licensed under [CC0 1.0 Universal](./LICENSE.md) (public domain).
