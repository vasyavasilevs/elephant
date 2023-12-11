#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <omp.h>
#include <time.h>

// #define NUM_THREADS 2
#define TAG_SIZE 16
#define BLOCK_SIZE 25
#define ELEPHANT_NUMBER_OF_BYTES 64
#define NONCE_NUM_BYTES 12
#define KEY_NUMBER_OF_BYTES 16
#define MAX_NUMBER_OF_KESSAK_ROUNDS 18
#define index(a, b) (((a) % 5) + 5 * ((b) % 5)) // Macro for calculating the index of an element in a state array by coordinates (a, b)

typedef unsigned char BYTE; // 8 bit
typedef unsigned long long LEN;
const unsigned int kessak_rho_offsets[25] = {0, 1, 6, 4, 3, 4, 4, 6, 7, 4, 3, 2, 3, 1, 7, 1, 5, 7, 5, 0, 2, 2, 5, 0, 6};

const BYTE kessak_round_constants[MAX_NUMBER_OF_KESSAK_ROUNDS] = {
    0x01, 0x82, 0x8a, 0x00, 0x8b, 0x01, 0x81, 0x09, 0x8a,
    0x88, 0x09, 0x0a, 0x8b, 0x8b, 0x89, 0x03, 0x02, 0x80};

// Macro to perform a cyclic left shift on eight bits
#define ROTATE_LEFT_8_BIT(a, offset) ((offset != 0) ? ((((BYTE)a) << offset) ^ (((BYTE)a) >> (sizeof(BYTE) * 8 - offset))) : a)

// Kessak part start___________________________________________________________________________________________________________

void theta(BYTE *TRIAL_ARR)
{
    unsigned int a, b;
    BYTE ARR[5], ARR2[5];

    for (a = 0; a < 5; a++)
    {
        ARR[a] = 0;
        for (b = 0; b < 5; b++)
            ARR[a] ^= TRIAL_ARR[index(a, b)];
    }

    for (a = 0; a < 5; a++)
        ARR2[a] = ROTATE_LEFT_8_BIT(ARR[(a + 1) % 5], 1) ^ ARR[(a + 4) % 5];

    for (a = 0; a < 5; a++)
        for (b = 0; b < 5; b++)
            TRIAL_ARR[index(a, b)] ^= ARR2[a];
}

void rho(BYTE *TRIAL_ARR)
{
    for (unsigned int a = 0; a < 5; a++)
        for (unsigned int b = 0; b < 5; b++)
            TRIAL_ARR[index(a, b)] = ROTATE_LEFT_8_BIT(TRIAL_ARR[index(a, b)], kessak_rho_offsets[index(a, b)]);
}

void pi(BYTE *TRIAL_ARR)
{
    BYTE fix_mask[25];

    for (unsigned int a = 0; a < 5; a++)
        for (unsigned int b = 0; b < 5; b++)
            fix_mask[index(a, b)] = TRIAL_ARR[index(a, b)];

    for (unsigned int a = 0; a < 5; a++)
        for (unsigned int b = 0; b < 5; b++)
            TRIAL_ARR[index(0 * a + 1 * b, 2 * a + 3 * b)] = fix_mask[index(a, b)];
}

void hi(BYTE *TRIAL_ARR)
{
    unsigned int a, b;
    BYTE ARR[5];

    for (b = 0; b < 5; b++)
    {
        for (a = 0; a < 5; a++)
            ARR[a] = TRIAL_ARR[index(a, b)] ^ ((~TRIAL_ARR[index(a + 1, b)]) & TRIAL_ARR[index(a + 2, b)]);
        for (a = 0; a < 5; a++)
            TRIAL_ARR[index(a, b)] = ARR[a];
    }
}

void io(BYTE *TRIAL_ARR, unsigned int ind)
{
    TRIAL_ARR[index(0, 0)] ^= kessak_round_constants[ind];
}

void Kessak200UnionForOneRound(BYTE *tmp_state, unsigned int tmp_round_idx)
{
    theta(tmp_state);
    rho(tmp_state);
    pi(tmp_state);
    hi(tmp_state);
    io(tmp_state, tmp_round_idx);
}

// The main function that applies a sequence of rounds (Kessak200UnionForOneRound) to complete one iteration of the KeccakP-200 algorithm
void permutation(BYTE *param)
{
    for (unsigned int t = 0; t < MAX_NUMBER_OF_KESSAK_ROUNDS; t++)
        Kessak200UnionForOneRound(param, t);
}

// Kessak part end___________________________________________________________________________________________________________

// Utils part start__________________________________________________________________________________________________________

// Left shift of byte
BYTE LeftShift(BYTE byte)
{
    return (byte << 1) | (byte >> 7);
}

// Cmp function compares two blocks of data byte by byte and returns 0 if the blocks are identical, and 1 otherwise
int cmp(const BYTE *a, const BYTE *b, LEN length)
{
    BYTE r = 0;
    // omp_set_num_threads(NUM_THREADS);
    // #pragma omp parallel for
    for (LEN i = 0; i < length; ++i)
        r |= a[i] ^ b[i];
    return r;
}

// Implements a single step linear feedback shifter (LFSR). Used to generate a mask for data encryption
void lfsr(BYTE *out, BYTE *in)
{
    BYTE cur = LeftShift(in[0]) ^ LeftShift(in[2]) ^ (in[13] << 1);
    // omp_set_num_threads(NUM_THREADS);
    // #pragma omp parallel for
    for (LEN t = 0; t < BLOCK_SIZE - 1; ++t)
        out[t] = in[t + 1];
    out[BLOCK_SIZE - 1] = cur;
}

void xorOfBlock(BYTE *tmp_state, const BYTE *block, LEN length)
{
    // omp_set_num_threads(NUM_THREADS);
    // #pragma omp parallel for
    for (LEN t = 0; t < length; ++t)
        tmp_state[t] ^= block[t];
}

// Utils part end_____________________________________________________________________________________________________________

// Unique part start__________________________________________________________________________________________________________

// Generate data blocks for processing associated data (AD) and ciphertext,
// respectively. Includes Nonce processing and adding padding if necessary

// Write the ith assocated data block to "output".
// The nonce is prepended and padding is added as required.
// adlen is the length of the associated data in bytes

// Both of these functions are used in the Elephant algorithm to preprocess
// the associated data and ciphertext blocks, adding additional blocks as needed.
// This may be necessary to ensure correct data alignment or to perform additional
// algorithm steps depending on the length of the input data

void get_associated_data_block(BYTE *out, const BYTE *aData, LEN len_aData, const BYTE *nonce, LEN t)
{
    const LEN offset = t * BLOCK_SIZE - (t != 0) * NONCE_NUM_BYTES;
    LEN len = 0;

    // First block contains nonce
    // Remark: nonce may not be longer then BLOCK_SIZE
    if (t == 0)
    {
        memcpy(out, nonce, NONCE_NUM_BYTES);
        len += NONCE_NUM_BYTES;
    }

    // If len_aData is divisible by BLOCK_SIZE, add an additional padding block
    if (t != 0 && offset == len_aData)
    {
        memset(out, 0x00, BLOCK_SIZE);
        out[0] = 0x01;
        return;
    }
    const LEN r_out = BLOCK_SIZE - len;
    const LEN r_data = len_aData - offset;

    // Fill with associated data if available
    if (r_out <= r_data)
    { // enough AD
        memcpy(out + len, aData + offset, r_out);
    }
    else
    {                   // not enough AD, need to pad
        if (r_data > 0) // ad might be nullptr
            memcpy(out + len, aData + offset, r_data);
        memset(out + len + r_data, 0x00, r_out - r_data);
        out[len + r_data] = 0x01;
    }
}

// Return the ith assocated data block.
// cipher_len is the length of the ciphertext in bytes
void get_ciphertext_block(BYTE *out, const BYTE *c, LEN cipher_len, LEN t)
{
    const LEN offset = t * BLOCK_SIZE;

    if (offset == cipher_len)
    {
        memset(out, 0x00, BLOCK_SIZE);
        out[0] = 0x01;
        return;
    }
    const LEN r_text = cipher_len - offset;

    if (BLOCK_SIZE <= r_text)
    {
        memcpy(out, c + offset, BLOCK_SIZE);
    }
    else
    {
        if (r_text > 0) // c might be nullptr
            memcpy(out, c + offset, r_text);
        memset(out + r_text, 0x00, BLOCK_SIZE - r_text);
        out[r_text] = 0x01;
    }
}

// The main function of authenticated encryption. Performs encryption or decryption depending on the encrypt flag

// Options:
// c: Ciphertext (output).
// tag: Authentication tag (output).
// m: Plaintext (input).
// len_message: Length of plaintext.
// aData: Associated data (input).
// len_aData: Length of the associated data.
// nonce: Nonce and plaintext (input).
// k: Key (entrance).
// encrypt: A flag indicating whether the operation is encryption (1) or decryption (0).

// Basic steps:

// 1. Calculating the number of blocks:
// nblocks_c, nblocks_m, nblocks_ad: Calculate the number of blocks for ciphertext, plaintext, and associated data.

// 2.Initializing the key and masks:
// expanded_key: The expanded key derived from the original key k using the permutation function.
// mask_buffer_1, mask_buffer_2, mask_buffer_3: Buffers for storing the mask_prev, mask_tmp and mask_next masks.
// previous_mask, current_mask, next_mask: Pointers to masks for the mask_tmp and mask_prev iterations.

// 3. Block processing cycle:
// Performed for each block of data (encryption or authentication step).
// Plaintext block encryption (m):
// Calculate the mask for the mask_next message (lfsr_step).
// Perform XOR operations to obtain an encrypted block.
// Update the authentication tag (tag_buffer) based on the ciphertext block and masks.
// Authentication of the ciphertext block (c) and associated data block (ad):
// Calculate the mask for the mask_next message (lfsr_step).
// Update authentication tag (tag_buffer) based on ciphertext/plaintext block and masks.
// Mask buffer offset:
// Update mask pointers for mask_next iteration.

// 4. Calculating the final authentication tag:
// Compute the final tag based on the extended key and tag_buffer.

// 5. Completion:
// Write the encrypted data and authentication tag to the appropriate output buffers.

// NOTE: NONCE_NUM_BYTES is a constant indicating the size of the Nonce (Number used ONCE)
// in bytes in the context of a cryptographic library or encryption algorithm. A Nonce
// is a random number that is used only once in combination with a key to generate
// unique encrypted messages

void 
implementation_of_crypto(
    BYTE *c, BYTE *tag, const BYTE *m, LEN len_message, const BYTE *aData, LEN len_aData,
    const BYTE *nonce, const BYTE *k, int encrypt) {

    const LEN num_of_blocks_cipher = 1 + len_message / BLOCK_SIZE;
    const LEN num_of_blocks_message = (len_message % BLOCK_SIZE) ? num_of_blocks_cipher : num_of_blocks_cipher - 1;
    const LEN num_of_blocks_adata = 1 + (NONCE_NUM_BYTES + len_aData) / BLOCK_SIZE;
    // num_of_blocks_it is used to denote a variable representing the number of iterations in the 
    // function's main loop. In the context of the Elephant algorithm, this is the 
    // number of blocks of data that need to be processed during the process of 
    // encrypting or decrypting a message
    const LEN num_of_blocks_it = (num_of_blocks_cipher > num_of_blocks_adata) ? num_of_blocks_cipher : num_of_blocks_adata + 1;

    BYTE expanded_k[BLOCK_SIZE] = {0};
    memcpy(expanded_k, k, KEY_NUMBER_OF_BYTES);
    permutation(expanded_k);

    // Buffers for storing mask_prev, mask_tmp and mask_next mask
    BYTE buffer_back[BLOCK_SIZE] = {0};
    BYTE buffer_current[BLOCK_SIZE] = {0};
    BYTE buffer_forward[BLOCK_SIZE] = {0};
    memcpy(buffer_current, expanded_k, BLOCK_SIZE);

    BYTE *mask_prev = buffer_back;
    BYTE *mask_tmp = buffer_current;
    BYTE *mask_next = buffer_forward;

    // Buffer to store current ciphertext/AD block
    BYTE elephant_buf[BLOCK_SIZE];

    // Tag buffer and initialization of tag to zero
    BYTE tag_buf[BLOCK_SIZE] = {0};
    memset(tag, 0, TAG_SIZE);

    LEN offset = 0;

    for (LEN t = 0; t < num_of_blocks_it; ++t) {
        // Compute mask for the next message
        lfsr(mask_next, mask_tmp);

        if (t < num_of_blocks_message) {
            memcpy(elephant_buf, nonce, NONCE_NUM_BYTES);
            memset(elephant_buf + NONCE_NUM_BYTES, 0, BLOCK_SIZE - NONCE_NUM_BYTES);
            xorOfBlock(elephant_buf, mask_tmp, BLOCK_SIZE);
            xorOfBlock(elephant_buf, mask_next, BLOCK_SIZE);
            permutation(elephant_buf);
            xorOfBlock(elephant_buf, mask_tmp, BLOCK_SIZE);
            xorOfBlock(elephant_buf, mask_next, BLOCK_SIZE);
            const LEN r_len = (t == num_of_blocks_message - 1) ? len_message - offset : BLOCK_SIZE;
            xorOfBlock(elephant_buf, m + offset, r_len);
            memcpy(c + offset, elephant_buf, r_len);
        }

        if (t > 0 && t <= num_of_blocks_cipher) {
            // Compute tag for ciphertext block
            get_ciphertext_block(tag_buf, encrypt ? c : m, len_message, t - 1);
            xorOfBlock(tag_buf, mask_prev, BLOCK_SIZE);
            xorOfBlock(tag_buf, mask_next, BLOCK_SIZE);
            permutation(tag_buf);
            xorOfBlock(tag_buf, mask_prev, BLOCK_SIZE);
            xorOfBlock(tag_buf, mask_next, BLOCK_SIZE);
            xorOfBlock(tag, tag_buf, TAG_SIZE);
        }

        // If there is any AD left, compute tag for AD block 
        if (t + 1 < num_of_blocks_adata) {
            get_associated_data_block(tag_buf, aData, len_aData, nonce, t + 1);
            xorOfBlock(tag_buf, mask_next, BLOCK_SIZE);
            permutation(tag_buf);
            xorOfBlock(tag_buf, mask_next, BLOCK_SIZE);
            xorOfBlock(tag, tag_buf, TAG_SIZE);
        }

        // Cyclically shift the mask buffers
        // Value of next_mask will be computed in the next iteration
        BYTE *const fix_mask = mask_prev;
        mask_prev = mask_tmp;
        mask_tmp = mask_next;
        mask_next = fix_mask;

        offset += BLOCK_SIZE;
    }
}

// c must be at least mlen + CRYPTO_ABYTES long
int 
encrypt(
    unsigned char *c, unsigned long long *len,
    const unsigned char *m, unsigned long long len_message,
    const unsigned char *aData, unsigned long long len_aData,
    const unsigned char *ns,
    const unsigned char *nonce,
    const unsigned char *k) {
    (void)ns;
    *len = len_message + TAG_SIZE;
    BYTE tag[TAG_SIZE];
    implementation_of_crypto(c, tag, m, len_message, aData, len_aData, nonce, k, 1);
    memcpy(c + len_message, tag, TAG_SIZE);
    return 0;
}

int 
decrypt(
    unsigned char *m, unsigned long long *len_message,
    unsigned char *ns,
    const unsigned char *c, unsigned long long len_cipher,
    const unsigned char *aData, unsigned long long len_aData,
    const unsigned char *nonce,
    const unsigned char *k) {
    (void)ns;
    if (len_cipher < TAG_SIZE)
        return -1;
    *len_message = len_cipher - TAG_SIZE;
    BYTE tag[TAG_SIZE];
    implementation_of_crypto(m, tag, c, *len_message, aData, len_aData, nonce, k, 0);
    return (cmp(c + *len_message, tag, TAG_SIZE) == 0) ? 0 : -1;
}

void 
string2hex(unsigned char *in, int len, char *out) {
    int turn;
    int t;
    t = 0;
    turn = 0;

    for (t = 0; t < len; t += 2) {
        sprintf((char *)(out + t), "%02X", in[turn]);
        turn += 1;
    }
    out[t++] = '\0';
}

void *
hex2byte(char *hex, unsigned char *bytes) {
    int t;
    int len = strlen(hex);

    for (t = 0; t < (len / 2); t++) {
        sscanf(hex + 2 * t, "%02hhx", &bytes[t]);
    }
}

int main(int argc, char *argv[]) {
    // start of measuring time
    struct timespec end, start;
    clock_gettime(CLOCK_MONOTONIC_RAW, &start);

    unsigned long long lenMessage;
    unsigned long long len;
    unsigned char plaintext[ELEPHANT_NUMBER_OF_BYTES];
    unsigned char cipher[ELEPHANT_NUMBER_OF_BYTES];
    unsigned char nonce[NONCE_NUM_BYTES] = "";
    unsigned char aData[TAG_SIZE] = "";
    unsigned char ns[TAG_SIZE] = "";
    unsigned char key[KEY_NUMBER_OF_BYTES];

    char pl[ELEPHANT_NUMBER_OF_BYTES] = "jopa";
    char hex[ELEPHANT_NUMBER_OF_BYTES] = "";
    char key_hex[2 * KEY_NUMBER_OF_BYTES + 1] = "0123456789ABCDEF0123456789ABCDEF";
    char nonce_hex[2 * NONCE_NUM_BYTES + 1] = "000000000000111111111111";
    char additional[TAG_SIZE] = "kek";

    strcpy(plaintext, pl);
    strcpy(aData, additional);
    hex2byte(key_hex, key);
    hex2byte(nonce_hex, nonce);

    printf("Elephant200v2 - Delirium!\n\n");
    printf("Plaintext: %s\n", plaintext);
    printf("Key: %s\n", key_hex);
    printf("Nonce: %s\n", nonce_hex);
    printf("Additional Data: %s\n\n", aData);

    printf("Plaintext: %s\n", plaintext);
    printf("Encryption started...\n");
    int err = encrypt(cipher, &len, plaintext, strlen(plaintext), aData, strlen(aData), ns, nonce, key);
    if (err != 0) {
        printf("Encryption failed\n");
        return EXIT_FAILURE;
    }

    string2hex(cipher, len, hex);

    printf("Cipher: %s. Length of the Cipher: %llu\n", hex, len);
    printf("Encryption done!\n\n");

    printf("Decryption started...\n");
    err = decrypt(plaintext, &lenMessage, ns, cipher, len, aData, strlen(aData), nonce, key);
    if (err != 0) {
        printf("Decryption failed\n");
        return EXIT_FAILURE;
    }

    printf("Plaintext: %s. Length of the Plaintext: %llu\n", plaintext, lenMessage);
    printf("Decryption done!\n\n");

    printf("Success!\n");

    // end of measuring time
    clock_gettime(CLOCK_MONOTONIC_RAW, &end);
    unsigned long int delta_us = (end.tv_sec - start.tv_sec) * 1000000 + (end.tv_nsec - start.tv_nsec) / 1000;
    printf("Elapsed time: %lu us\n", delta_us);

    return EXIT_SUCCESS;
}