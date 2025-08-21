use anyhow::{Error, Result};
use lazy_static::lazy_static;
use std::collections::HashMap;
use std::fmt::Display;
use std::str::FromStr;

lazy_static! {
    static ref COLORS: HashMap<String, String> = {
        HashMap::from([
            ("b".to_string(), "#0000FF".to_string()),
            ("g".to_string(), "#008000".to_string()),
            ("r".to_string(), "#FF0000".to_string()),
            ("c".to_string(), "#00BFBF".to_string()),
            ("m".to_string(), "#BF00BF".to_string()),
            ("y".to_string(), "#BFBF00".to_string()),
            ("k".to_string(), "#000000".to_string()),
            ("w".to_string(), "#FFFFFF".to_string()),
            ("aliceblue".to_string(), "#F0F8FF".to_string()),
            ("antiquewhite".to_string(), "#FAEBD7".to_string()),
            ("aqua".to_string(), "#00FFFF".to_string()),
            ("aquamarine".to_string(), "#7FFFD4".to_string()),
            ("azure".to_string(), "#F0FFFF".to_string()),
            ("beige".to_string(), "#F5F5DC".to_string()),
            ("bisque".to_string(), "#FFE4C4".to_string()),
            ("black".to_string(), "#000000".to_string()),
            ("blanchedalmond".to_string(), "#FFEBCD".to_string()),
            ("blue".to_string(), "#0000FF".to_string()),
            ("blueviolet".to_string(), "#8A2BE2".to_string()),
            ("brown".to_string(), "#A52A2A".to_string()),
            ("burlywood".to_string(), "#DEB887".to_string()),
            ("cadetblue".to_string(), "#5F9EA0".to_string()),
            ("chartreuse".to_string(), "#7FFF00".to_string()),
            ("chocolate".to_string(), "#D2691E".to_string()),
            ("coral".to_string(), "#FF7F50".to_string()),
            ("cornflowerblue".to_string(), "#6495ED".to_string()),
            ("cornsilk".to_string(), "#FFF8DC".to_string()),
            ("crimson".to_string(), "#DC143C".to_string()),
            ("cyan".to_string(), "#00FFFF".to_string()),
            ("darkblue".to_string(), "#00008B".to_string()),
            ("darkcyan".to_string(), "#008B8B".to_string()),
            ("darkgoldenrod".to_string(), "#B8860B".to_string()),
            ("darkgray".to_string(), "#A9A9A9".to_string()),
            ("darkgreen".to_string(), "#006400".to_string()),
            ("darkgrey".to_string(), "#A9A9A9".to_string()),
            ("darkkhaki".to_string(), "#BDB76B".to_string()),
            ("darkmagenta".to_string(), "#8B008B".to_string()),
            ("darkolivegreen".to_string(), "#556B2F".to_string()),
            ("darkorange".to_string(), "#FF8C00".to_string()),
            ("darkorchid".to_string(), "#9932CC".to_string()),
            ("darkred".to_string(), "#8B0000".to_string()),
            ("darksalmon".to_string(), "#E9967A".to_string()),
            ("darkseagreen".to_string(), "#8FBC8F".to_string()),
            ("darkslateblue".to_string(), "#483D8B".to_string()),
            ("darkslategray".to_string(), "#2F4F4F".to_string()),
            ("darkslategrey".to_string(), "#2F4F4F".to_string()),
            ("darkturquoise".to_string(), "#00CED1".to_string()),
            ("darkviolet".to_string(), "#9400D3".to_string()),
            ("deeppink".to_string(), "#FF1493".to_string()),
            ("deepskyblue".to_string(), "#00BFFF".to_string()),
            ("dimgray".to_string(), "#696969".to_string()),
            ("dimgrey".to_string(), "#696969".to_string()),
            ("dodgerblue".to_string(), "#1E90FF".to_string()),
            ("firebrick".to_string(), "#B22222".to_string()),
            ("floralwhite".to_string(), "#FFFAF0".to_string()),
            ("forestgreen".to_string(), "#228B22".to_string()),
            ("fuchsia".to_string(), "#FF00FF".to_string()),
            ("gainsboro".to_string(), "#DCDCDC".to_string()),
            ("ghostwhite".to_string(), "#F8F8FF".to_string()),
            ("gold".to_string(), "#FFD700".to_string()),
            ("goldenrod".to_string(), "#DAA520".to_string()),
            ("gray".to_string(), "#808080".to_string()),
            ("green".to_string(), "#008000".to_string()),
            ("greenyellow".to_string(), "#ADFF2F".to_string()),
            ("grey".to_string(), "#808080".to_string()),
            ("honeydew".to_string(), "#F0FFF0".to_string()),
            ("hotpink".to_string(), "#FF69B4".to_string()),
            ("indianred".to_string(), "#CD5C5C".to_string()),
            ("indigo".to_string(), "#4B0082".to_string()),
            ("ivory".to_string(), "#FFFFF0".to_string()),
            ("khaki".to_string(), "#F0E68C".to_string()),
            ("lavender".to_string(), "#E6E6FA".to_string()),
            ("lavenderblush".to_string(), "#FFF0F5".to_string()),
            ("lawngreen".to_string(), "#7CFC00".to_string()),
            ("lemonchiffon".to_string(), "#FFFACD".to_string()),
            ("lightblue".to_string(), "#ADD8E6".to_string()),
            ("lightcoral".to_string(), "#F08080".to_string()),
            ("lightcyan".to_string(), "#E0FFFF".to_string()),
            ("lightgoldenrodyellow".to_string(), "#FAFAD2".to_string()),
            ("lightgray".to_string(), "#D3D3D3".to_string()),
            ("lightgreen".to_string(), "#90EE90".to_string()),
            ("lightgrey".to_string(), "#D3D3D3".to_string()),
            ("lightpink".to_string(), "#FFB6C1".to_string()),
            ("lightsalmon".to_string(), "#FFA07A".to_string()),
            ("lightseagreen".to_string(), "#20B2AA".to_string()),
            ("lightskyblue".to_string(), "#87CEFA".to_string()),
            ("lightslategray".to_string(), "#778899".to_string()),
            ("lightslategrey".to_string(), "#778899".to_string()),
            ("lightsteelblue".to_string(), "#B0C4DE".to_string()),
            ("lightyellow".to_string(), "#FFFFE0".to_string()),
            ("lime".to_string(), "#00FF00".to_string()),
            ("limegreen".to_string(), "#32CD32".to_string()),
            ("linen".to_string(), "#FAF0E6".to_string()),
            ("magenta".to_string(), "#FF00FF".to_string()),
            ("maroon".to_string(), "#800000".to_string()),
            ("mediumaquamarine".to_string(), "#66CDAA".to_string()),
            ("mediumblue".to_string(), "#0000CD".to_string()),
            ("mediumorchid".to_string(), "#BA55D3".to_string()),
            ("mediumpurple".to_string(), "#9370DB".to_string()),
            ("mediumseagreen".to_string(), "#3CB371".to_string()),
            ("mediumslateblue".to_string(), "#7B68EE".to_string()),
            ("mediumspringgreen".to_string(), "#00FA9A".to_string()),
            ("mediumturquoise".to_string(), "#48D1CC".to_string()),
            ("mediumvioletred".to_string(), "#C71585".to_string()),
            ("midnightblue".to_string(), "#191970".to_string()),
            ("mintcream".to_string(), "#F5FFFA".to_string()),
            ("mistyrose".to_string(), "#FFE4E1".to_string()),
            ("moccasin".to_string(), "#FFE4B5".to_string()),
            ("navajowhite".to_string(), "#FFDEAD".to_string()),
            ("navy".to_string(), "#000080".to_string()),
            ("oldlace".to_string(), "#FDF5E6".to_string()),
            ("olive".to_string(), "#808000".to_string()),
            ("olivedrab".to_string(), "#6B8E23".to_string()),
            ("orange".to_string(), "#FFA500".to_string()),
            ("orangered".to_string(), "#FF4500".to_string()),
            ("orchid".to_string(), "#DA70D6".to_string()),
            ("palegoldenrod".to_string(), "#EEE8AA".to_string()),
            ("palegreen".to_string(), "#98FB98".to_string()),
            ("paleturquoise".to_string(), "#AFEEEE".to_string()),
            ("palevioletred".to_string(), "#DB7093".to_string()),
            ("papayawhip".to_string(), "#FFEFD5".to_string()),
            ("peachpuff".to_string(), "#FFDAB9".to_string()),
            ("peru".to_string(), "#CD853F".to_string()),
            ("pink".to_string(), "#FFC0CB".to_string()),
            ("plum".to_string(), "#DDA0DD".to_string()),
            ("powderblue".to_string(), "#B0E0E6".to_string()),
            ("purple".to_string(), "#800080".to_string()),
            ("rebeccapurple".to_string(), "#663399".to_string()),
            ("red".to_string(), "#FF0000".to_string()),
            ("rosybrown".to_string(), "#BC8F8F".to_string()),
            ("royalblue".to_string(), "#4169E1".to_string()),
            ("saddlebrown".to_string(), "#8B4513".to_string()),
            ("salmon".to_string(), "#FA8072".to_string()),
            ("sandybrown".to_string(), "#F4A460".to_string()),
            ("seagreen".to_string(), "#2E8B57".to_string()),
            ("seashell".to_string(), "#FFF5EE".to_string()),
            ("sienna".to_string(), "#A0522D".to_string()),
            ("silver".to_string(), "#C0C0C0".to_string()),
            ("skyblue".to_string(), "#87CEEB".to_string()),
            ("slateblue".to_string(), "#6A5ACD".to_string()),
            ("slategray".to_string(), "#708090".to_string()),
            ("slategrey".to_string(), "#708090".to_string()),
            ("snow".to_string(), "#FFFAFA".to_string()),
            ("springgreen".to_string(), "#00FF7F".to_string()),
            ("steelblue".to_string(), "#4682B4".to_string()),
            ("tan".to_string(), "#D2B48C".to_string()),
            ("teal".to_string(), "#008080".to_string()),
            ("thistle".to_string(), "#D8BFD8".to_string()),
            ("tomato".to_string(), "#FF6347".to_string()),
            ("turquoise".to_string(), "#40E0D0".to_string()),
            ("violet".to_string(), "#EE82EE".to_string()),
            ("wheat".to_string(), "#F5DEB3".to_string()),
            ("white".to_string(), "#FFFFFF".to_string()),
            ("whitesmoke".to_string(), "#F5F5F5".to_string()),
            ("yellow".to_string(), "#FFFF00".to_string()),
            ("yellowgreen".to_string(), "#9ACD32".to_string()),
        ])
    };
}

#[derive(Clone, Debug)]
pub struct Color {
    r: u8,
    g: u8,
    b: u8,
}

impl FromStr for Color {
    type Err = Error;

    fn from_str(s: &str) -> Result<Self> {
        let s = if !s.starts_with("#") {
            if let Some(s) = COLORS.get(s) {
                s
            } else {
                return Err(Error::msg(format!("invalid color: {}", s)));
            }
        } else {
            s
        };
        let r = u8::from_str_radix(&s[1..3], 16)?;
        let g = u8::from_str_radix(&s[3..5], 16)?;
        let b = u8::from_str_radix(&s[5..], 16)?;
        Ok(Self { r, g, b })
    }
}

impl Display for Color {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "#{:02X}{:02X}{:02X}", self.r, self.g, self.b)
    }
}

impl Color {
    pub fn to_rgb(&self) -> Vec<u8> {
        vec![self.r, self.g, self.b]
    }
}
