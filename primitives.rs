//! Kona primitive registry.
//!
//! This mirrors the primitive table in `kona-master/src/k.c`. The `status`
//! field is intentionally blunt so the Rust port can grow by filling in one
//! primitive at a time while keeping the full surface visible.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrimitiveStatus {
    Implemented,
    Partial,
    Pending,
    OutOfCore,
}

#[allow(dead_code)]
#[derive(Debug, Clone, Copy)]
pub struct Primitive {
    pub name: &'static str,
    pub valence: u8,
    pub c_function: &'static str,
    pub status: PrimitiveStatus,
}

pub const PRIMITIVES: &[Primitive] = &[
    Primitive {
        name: "/",
        valence: 0,
        c_function: "over",
        status: PrimitiveStatus::Implemented,
    },
    Primitive {
        name: "\\",
        valence: 0,
        c_function: "scan",
        status: PrimitiveStatus::Implemented,
    },
    Primitive {
        name: "'",
        valence: 0,
        c_function: "each",
        status: PrimitiveStatus::Implemented,
    },
    Primitive {
        name: "/:",
        valence: 0,
        c_function: "eachright",
        status: PrimitiveStatus::Pending,
    },
    Primitive {
        name: "\\:",
        valence: 0,
        c_function: "eachleft",
        status: PrimitiveStatus::Pending,
    },
    Primitive {
        name: "':",
        valence: 0,
        c_function: "eachpair",
        status: PrimitiveStatus::Pending,
    },
    Primitive {
        name: "+",
        valence: 1,
        c_function: "flip",
        status: PrimitiveStatus::Implemented,
    },
    Primitive {
        name: "+",
        valence: 2,
        c_function: "plus",
        status: PrimitiveStatus::Implemented,
    },
    Primitive {
        name: "-",
        valence: 1,
        c_function: "negate",
        status: PrimitiveStatus::Implemented,
    },
    Primitive {
        name: "-",
        valence: 2,
        c_function: "minus",
        status: PrimitiveStatus::Implemented,
    },
    Primitive {
        name: "*",
        valence: 1,
        c_function: "first",
        status: PrimitiveStatus::Implemented,
    },
    Primitive {
        name: "*",
        valence: 2,
        c_function: "times",
        status: PrimitiveStatus::Implemented,
    },
    Primitive {
        name: "%",
        valence: 1,
        c_function: "reciprocal",
        status: PrimitiveStatus::Implemented,
    },
    Primitive {
        name: "%",
        valence: 2,
        c_function: "divide",
        status: PrimitiveStatus::Implemented,
    },
    Primitive {
        name: "|",
        valence: 1,
        c_function: "reverse",
        status: PrimitiveStatus::Implemented,
    },
    Primitive {
        name: "|",
        valence: 2,
        c_function: "max_or",
        status: PrimitiveStatus::Implemented,
    },
    Primitive {
        name: "&",
        valence: 1,
        c_function: "where",
        status: PrimitiveStatus::Partial,
    },
    Primitive {
        name: "&",
        valence: 2,
        c_function: "min_and",
        status: PrimitiveStatus::Implemented,
    },
    Primitive {
        name: "^",
        valence: 1,
        c_function: "shape",
        status: PrimitiveStatus::Pending,
    },
    Primitive {
        name: "^",
        valence: 2,
        c_function: "power",
        status: PrimitiveStatus::Implemented,
    },
    Primitive {
        name: "!",
        valence: 1,
        c_function: "enumerate",
        status: PrimitiveStatus::Pending,
    },
    Primitive {
        name: "!",
        valence: 2,
        c_function: "rotate_mod",
        status: PrimitiveStatus::Partial,
    },
    Primitive {
        name: "<",
        valence: 1,
        c_function: "grade_up",
        status: PrimitiveStatus::Implemented,
    },
    Primitive {
        name: "<",
        valence: 2,
        c_function: "less",
        status: PrimitiveStatus::Implemented,
    },
    Primitive {
        name: ">",
        valence: 1,
        c_function: "grade_down",
        status: PrimitiveStatus::Implemented,
    },
    Primitive {
        name: ">",
        valence: 2,
        c_function: "more",
        status: PrimitiveStatus::Implemented,
    },
    Primitive {
        name: "=",
        valence: 1,
        c_function: "group",
        status: PrimitiveStatus::Pending,
    },
    Primitive {
        name: "=",
        valence: 2,
        c_function: "equals",
        status: PrimitiveStatus::Implemented,
    },
    Primitive {
        name: "~",
        valence: 1,
        c_function: "not_attribute",
        status: PrimitiveStatus::Pending,
    },
    Primitive {
        name: "~",
        valence: 2,
        c_function: "match",
        status: PrimitiveStatus::Implemented,
    },
    Primitive {
        name: "@",
        valence: 1,
        c_function: "atom",
        status: PrimitiveStatus::Pending,
    },
    Primitive {
        name: "@",
        valence: 2,
        c_function: "at",
        status: PrimitiveStatus::Pending,
    },
    Primitive {
        name: "?",
        valence: 1,
        c_function: "range",
        status: PrimitiveStatus::Implemented,
    },
    Primitive {
        name: "?",
        valence: 2,
        c_function: "what",
        status: PrimitiveStatus::Pending,
    },
    Primitive {
        name: "_",
        valence: 1,
        c_function: "floor_verb",
        status: PrimitiveStatus::Implemented,
    },
    Primitive {
        name: "_",
        valence: 2,
        c_function: "drop_cut",
        status: PrimitiveStatus::Pending,
    },
    Primitive {
        name: ",",
        valence: 1,
        c_function: "enlist",
        status: PrimitiveStatus::Pending,
    },
    Primitive {
        name: ",",
        valence: 2,
        c_function: "join",
        status: PrimitiveStatus::Pending,
    },
    Primitive {
        name: "#",
        valence: 1,
        c_function: "count",
        status: PrimitiveStatus::Pending,
    },
    Primitive {
        name: "#",
        valence: 2,
        c_function: "take_reshape",
        status: PrimitiveStatus::Implemented,
    },
    Primitive {
        name: "$",
        valence: 1,
        c_function: "format",
        status: PrimitiveStatus::Pending,
    },
    Primitive {
        name: "$",
        valence: 2,
        c_function: "dollar",
        status: PrimitiveStatus::Pending,
    },
    Primitive {
        name: ".",
        valence: 1,
        c_function: "dot_monadic",
        status: PrimitiveStatus::Pending,
    },
    Primitive {
        name: ".",
        valence: 2,
        c_function: "dot",
        status: PrimitiveStatus::Pending,
    },
    Primitive {
        name: ":",
        valence: 1,
        c_function: "colon_monadic",
        status: PrimitiveStatus::Pending,
    },
    Primitive {
        name: ":",
        valence: 2,
        c_function: "colon_dyadic",
        status: PrimitiveStatus::Pending,
    },
    Primitive {
        name: "_dot",
        valence: 2,
        c_function: "_dot",
        status: PrimitiveStatus::Implemented,
    },
    Primitive {
        name: "_mul",
        valence: 2,
        c_function: "_mul",
        status: PrimitiveStatus::Partial,
    },
    Primitive {
        name: "_pmul",
        valence: 2,
        c_function: "_pmul",
        status: PrimitiveStatus::Partial,
    },
    Primitive {
        name: "_sqrt",
        valence: 1,
        c_function: "_sqrt",
        status: PrimitiveStatus::Implemented,
    },
    Primitive {
        name: "_exp",
        valence: 1,
        c_function: "_exp",
        status: PrimitiveStatus::Implemented,
    },
    Primitive {
        name: "_floor",
        valence: 1,
        c_function: "_floor",
        status: PrimitiveStatus::Implemented,
    },
    Primitive {
        name: "_ceil",
        valence: 1,
        c_function: "_ceil",
        status: PrimitiveStatus::Implemented,
    },
    Primitive {
        name: "_abs",
        valence: 1,
        c_function: "_abs",
        status: PrimitiveStatus::Pending,
    },
    Primitive {
        name: "0:",
        valence: 1,
        c_function: "_0m",
        status: PrimitiveStatus::OutOfCore,
    },
    Primitive {
        name: "0:",
        valence: 2,
        c_function: "_0d",
        status: PrimitiveStatus::OutOfCore,
    },
    Primitive {
        name: "1:",
        valence: 1,
        c_function: "_1m",
        status: PrimitiveStatus::OutOfCore,
    },
    Primitive {
        name: "1:",
        valence: 2,
        c_function: "_1d",
        status: PrimitiveStatus::OutOfCore,
    },
    Primitive {
        name: "2:",
        valence: 1,
        c_function: "_2m",
        status: PrimitiveStatus::Implemented,
    },
    Primitive {
        name: "2:",
        valence: 2,
        c_function: "_2d",
        status: PrimitiveStatus::Implemented,
    },
    Primitive {
        name: "3:",
        valence: 1,
        c_function: "_3m",
        status: PrimitiveStatus::OutOfCore,
    },
    Primitive {
        name: "3:",
        valence: 2,
        c_function: "_3d",
        status: PrimitiveStatus::OutOfCore,
    },
    Primitive {
        name: "4:",
        valence: 1,
        c_function: "_4m",
        status: PrimitiveStatus::OutOfCore,
    },
    Primitive {
        name: "4:",
        valence: 2,
        c_function: "_4d",
        status: PrimitiveStatus::OutOfCore,
    },
    Primitive {
        name: "5:",
        valence: 1,
        c_function: "_5m",
        status: PrimitiveStatus::OutOfCore,
    },
    Primitive {
        name: "5:",
        valence: 2,
        c_function: "_5d",
        status: PrimitiveStatus::OutOfCore,
    },
    Primitive {
        name: "6:",
        valence: 1,
        c_function: "_6m",
        status: PrimitiveStatus::OutOfCore,
    },
    Primitive {
        name: "6:",
        valence: 2,
        c_function: "_6d",
        status: PrimitiveStatus::OutOfCore,
    },
];

pub fn implemented_count() -> usize {
    PRIMITIVES
        .iter()
        .filter(|p| p.status == PrimitiveStatus::Implemented)
        .count()
}
