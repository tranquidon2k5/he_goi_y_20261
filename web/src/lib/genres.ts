// Thể loại MovieLens (lưu nguyên tiếng Anh trong CSDL) → nhãn hiển thị tiếng Việt
export const GENRE_VI: Record<string, string> = {
  Action: "Hành động",
  Adventure: "Phiêu lưu",
  Animation: "Hoạt hình",
  Children: "Thiếu nhi",
  Comedy: "Hài",
  Crime: "Tội phạm",
  Documentary: "Tài liệu",
  Drama: "Chính kịch",
  Fantasy: "Giả tưởng",
  "Film-Noir": "Phim đen",
  Horror: "Kinh dị",
  IMAX: "IMAX",
  Musical: "Nhạc kịch",
  Mystery: "Bí ẩn",
  Romance: "Lãng mạn",
  "Sci-Fi": "Khoa học viễn tưởng",
  Thriller: "Giật gân",
  War: "Chiến tranh",
  Western: "Viễn Tây",
};

export const GLYPH: Record<string, string> = {
  Action: "✦", Adventure: "⛰", Animation: "✿", Children: "☻", Comedy: "☺", Crime: "♠", Documentary: "◎",
  Drama: "◐", Fantasy: "✧", "Film-Noir": "◆", Horror: "☾", IMAX: "▣", Musical: "♪", Mystery: "?",
  Romance: "♥", "Sci-Fi": "◉", Thriller: "⚡", War: "✠", Western: "☼",
};

export const gvi = (g: string) => GENRE_VI[g] ?? g;

export function rowTitle(title: string, kind: string): string {
  return kind === "genre" ? gvi(title) : title;
}
