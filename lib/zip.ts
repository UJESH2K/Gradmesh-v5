/**
 * A small ZIP writer for the browser, for uploading a dataset folder.
 *
 * It exists for a Mac habit: Safari unzips a downloaded archive by default, so
 * a dataset someone just downloaded is usually a folder by the time they reach
 * the Datasets page, and the coordinator takes zips. Entries are stored rather
 * than compressed, because images are compressed already, so the only work is
 * a CRC per file. The result is a Blob that references the files rather than
 * copying them, so even a large dataset costs little memory. ZIP64 records are
 * written once the archive passes 4 GB or 65,535 entries, which Python's
 * zipfile reads.
 */

export type ZipEntry = { path: string; file: File };

const CRC_TABLE = (() => {
  const table = new Uint32Array(256);
  for (let n = 0; n < 256; n += 1) {
    let c = n;
    for (let k = 0; k < 8; k += 1) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
    table[n] = c >>> 0;
  }
  return table;
})();

function crc32(bytes: Uint8Array): number {
  let crc = 0xffffffff;
  for (let index = 0; index < bytes.length; index += 1) {
    crc = CRC_TABLE[(crc ^ bytes[index]) & 0xff] ^ (crc >>> 8);
  }
  return (crc ^ 0xffffffff) >>> 0;
}

function dosTime(ms: number): { time: number; date: number } {
  const when = new Date(ms || Date.now());
  const year = Math.max(1980, when.getFullYear());
  return {
    time: (when.getHours() << 11) | (when.getMinutes() << 5) | Math.floor(when.getSeconds() / 2),
    date: ((year - 1980) << 9) | ((when.getMonth() + 1) << 5) | when.getDate(),
  };
}

const MAX32 = 0xffffffff;
const encoder = new TextEncoder();

/** Files an operating system adds that are not part of a dataset. */
export function isOsJunk(path: string): boolean {
  const parts = path.split("/");
  const name = parts[parts.length - 1] || "";
  return (
    parts.includes("__MACOSX") ||
    name.startsWith("._") ||
    name === ".DS_Store" ||
    name === "Thumbs.db" ||
    name === "desktop.ini"
  );
}

export async function zipEntries(
  entries: ZipEntry[],
  onProgress?: (doneBytes: number, totalBytes: number) => void
): Promise<Blob> {
  const totalBytes = entries.reduce((sum, entry) => sum + entry.file.size, 0);
  const parts: BlobPart[] = [];
  const central: Uint8Array[] = [];
  let offset = 0;
  let done = 0;

  for (const { path, file } of entries) {
    if (file.size >= MAX32) throw new Error(`${path} is larger than 4 GB, which a dataset image never is.`);
    const name = encoder.encode(path);
    const data = new Uint8Array(await file.arrayBuffer());
    const crc = crc32(data);
    const { time, date } = dosTime(file.lastModified);

    const local = new DataView(new ArrayBuffer(30));
    local.setUint32(0, 0x04034b50, true);
    local.setUint16(4, 20, true); // version needed
    local.setUint16(6, 0x0800, true); // UTF-8 names
    local.setUint16(8, 0, true); // stored
    local.setUint16(10, time, true);
    local.setUint16(12, date, true);
    local.setUint32(14, crc, true);
    local.setUint32(18, file.size, true);
    local.setUint32(22, file.size, true);
    local.setUint16(26, name.length, true);
    local.setUint16(28, 0, true);
    parts.push(new Uint8Array(local.buffer), name, file);

    // The central record points back at the local header; past 4 GB that
    // offset needs a ZIP64 extra field.
    const far = offset >= MAX32;
    const record = new DataView(new ArrayBuffer(46 + (far ? 12 : 0)));
    record.setUint32(0, 0x02014b50, true);
    record.setUint16(4, far ? 45 : 20, true); // version made by
    record.setUint16(6, far ? 45 : 20, true); // version needed
    record.setUint16(8, 0x0800, true);
    record.setUint16(10, 0, true);
    record.setUint16(12, time, true);
    record.setUint16(14, date, true);
    record.setUint32(16, crc, true);
    record.setUint32(20, file.size, true);
    record.setUint32(24, file.size, true);
    record.setUint16(28, name.length, true);
    record.setUint16(30, far ? 12 : 0, true);
    record.setUint32(42, far ? MAX32 : offset, true);
    const header = new Uint8Array(record.buffer, 0, 46);
    const combined = new Uint8Array(46 + name.length + (far ? 12 : 0));
    combined.set(header, 0);
    combined.set(name, 46);
    if (far) {
      const extra = new DataView(combined.buffer, 46 + name.length, 12);
      extra.setUint16(0, 0x0001, true);
      extra.setUint16(2, 8, true);
      extra.setBigUint64(4, BigInt(offset), true);
    }
    central.push(combined);

    offset += 30 + name.length + file.size;
    done += file.size;
    onProgress?.(done, totalBytes);
  }

  const directoryOffset = offset;
  const directorySize = central.reduce((sum, record) => sum + record.length, 0);
  parts.push(...central);

  const count = entries.length;
  const zip64 = count > 0xffff || directoryOffset >= MAX32 || directorySize >= MAX32;
  if (zip64) {
    const record = new DataView(new ArrayBuffer(56 + 20));
    record.setUint32(0, 0x06064b50, true);
    record.setBigUint64(4, BigInt(44), true);
    record.setUint16(12, 45, true);
    record.setUint16(14, 45, true);
    record.setUint32(16, 0, true);
    record.setUint32(20, 0, true);
    record.setBigUint64(24, BigInt(count), true);
    record.setBigUint64(32, BigInt(count), true);
    record.setBigUint64(40, BigInt(directorySize), true);
    record.setBigUint64(48, BigInt(directoryOffset), true);
    // Locator: where the ZIP64 end record starts.
    record.setUint32(56, 0x07064b50, true);
    record.setUint32(60, 0, true);
    record.setBigUint64(64, BigInt(directoryOffset + directorySize), true);
    record.setUint32(72, 1, true);
    parts.push(new Uint8Array(record.buffer));
  }

  const end = new DataView(new ArrayBuffer(22));
  end.setUint32(0, 0x06054b50, true);
  end.setUint16(8, zip64 ? 0xffff : count, true);
  end.setUint16(10, zip64 ? 0xffff : count, true);
  end.setUint32(12, zip64 ? MAX32 : directorySize, true);
  end.setUint32(16, zip64 ? MAX32 : directoryOffset, true);
  parts.push(new Uint8Array(end.buffer));

  return new Blob(parts, { type: "application/zip" });
}

/**
 * Every file under the folders (or files) dropped on the page, with paths
 * relative to what was dropped. Takes the entries, not the DataTransfer:
 * a DataTransfer is emptied once the drop handler returns, so the caller has
 * to read `webkitGetAsEntry()` synchronously and pass the result here.
 */
export async function filesFromEntries(roots: FileSystemEntry[]): Promise<ZipEntry[]> {
  const found: ZipEntry[] = [];

  const readAll = async (reader: FileSystemDirectoryReader) => {
    const all: FileSystemEntry[] = [];
    // readEntries returns at most a batch at a time; an empty batch means done.
    for (;;) {
      const batch = await new Promise<FileSystemEntry[]>((resolve, reject) => reader.readEntries(resolve, reject));
      if (batch.length === 0) return all;
      all.push(...batch);
    }
  };

  const walk = async (entry: FileSystemEntry, prefix: string): Promise<void> => {
    const path = prefix + entry.name;
    if (isOsJunk(path)) return;
    if (entry.isFile) {
      const file = await new Promise<File>((resolve, reject) => (entry as FileSystemFileEntry).file(resolve, reject));
      found.push({ path, file });
    } else if (entry.isDirectory) {
      for (const child of await readAll((entry as FileSystemDirectoryEntry).createReader())) {
        await walk(child, path + "/");
      }
    }
  };

  for (const root of roots) await walk(root, "");
  return found;
}

/** A chosen folder (an `<input webkitdirectory>`) as zip entries. */
export function filesFromFolderInput(files: FileList): ZipEntry[] {
  return Array.from(files)
    .map((file) => ({ path: file.webkitRelativePath || file.name, file }))
    .filter((entry) => !isOsJunk(entry.path));
}
