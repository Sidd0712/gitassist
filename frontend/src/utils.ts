export function humanize(value: string): string {
  const words = value.split('_');
  return words.map((word, i) => (i === 0 ? word.charAt(0).toUpperCase() + word.slice(1) : word)).join('-');
}
