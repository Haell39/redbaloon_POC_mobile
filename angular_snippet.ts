// ==================================================================
// angular_snippet.ts
// Exemplo de integração Angular para a equipe de frontend.
// Copie e adapte os trechos abaixo no seu projeto Angular.
// ==================================================================

// ─── 1. face-verify.service.ts ───────────────────────────────────

import { Injectable } from "@angular/core";
import { HttpClient } from "@angular/common/http";
import { Observable } from "rxjs";

/** Contrato de resposta do backend /verify */
export interface VerifyResponse {
  match: boolean;
  name: string;
  confidence: number;
  message: string;
}

@Injectable({ providedIn: "root" })
export class FaceVerifyService {
  // Ajuste a URL para o IP do servidor FastAPI
  private readonly apiUrl = "http://192.168.0.100:8000";

  constructor(private http: HttpClient) {}

  /**
   * Envia selfie (Blob JPEG) para validação facial.
   * Retorna Observable com o resultado da verificação.
   */
  verify(imageBlob: Blob): Observable<VerifyResponse> {
    const formData = new FormData();
    formData.append("file", imageBlob, "selfie.jpg");
    return this.http.post<VerifyResponse>(`${this.apiUrl}/verify`, formData);
  }

  /** Lista os nomes de rostos cadastrados (debug). */
  getUsers(): Observable<{ users: string[] }> {
    return this.http.get<{ users: string[] }>(`${this.apiUrl}/users`);
  }

  /** Força releitura da pasta database/ no servidor. */
  refreshDatabase(): Observable<{
    status: string;
    message: string;
    users: string[];
  }> {
    return this.http.get<{ status: string; message: string; users: string[] }>(
      `${this.apiUrl}/refresh-db`,
    );
  }
}

// ─── 2. selfie.component.ts ─────────────────────────────────────

import {
  Component,
  ElementRef,
  OnDestroy,
  OnInit,
  ViewChild,
} from "@angular/core";
// import { FaceVerifyService, VerifyResponse } from '../services/face-verify.service';

@Component({
  selector: "app-selfie",
  template: `
    <div class="selfie-container">
      <!-- Camera preview -->
      <video #videoEl autoplay playsinline muted class="camera-preview"></video>

      <!-- Hidden canvas for frame capture -->
      <canvas #canvasEl style="display:none"></canvas>

      <!-- Result -->
      <div *ngIf="result" class="result-card" [ngClass]="resultTier">
        <p class="result-name">{{ result.name }}</p>
        <p class="result-confidence">
          {{ (result.confidence * 100).toFixed(1) }}%
        </p>
        <p class="result-message">{{ result.message }}</p>
      </div>

      <!-- Action -->
      <button
        (click)="onVerify()"
        [disabled]="loading || !cameraReady"
        class="btn-verify"
      >
        {{ loading ? "Verificando…" : "📸 VALIDAR ACESSO" }}
      </button>
    </div>
  `,
  styles: [
    `
      .selfie-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        gap: 16px;
        padding: 16px;
      }
      .camera-preview {
        width: 100%;
        max-width: 480px;
        border-radius: 16px;
        transform: scaleX(-1);
        background: #000;
      }
      .result-card {
        padding: 16px;
        border-radius: 12px;
        text-align: center;
        width: 100%;
        max-width: 480px;
      }
      .result-card.success {
        background: rgba(0, 184, 148, 0.15);
        border: 2px solid #00b894;
      }
      .result-card.warning {
        background: rgba(253, 203, 110, 0.15);
        border: 2px solid #fdcb6e;
      }
      .result-card.error {
        background: rgba(214, 48, 49, 0.15);
        border: 2px solid #d63031;
      }
      .result-name {
        font-size: 1.2rem;
        font-weight: 700;
      }
      .result-confidence {
        font-size: 0.9rem;
        opacity: 0.7;
      }
      .result-message {
        font-size: 0.8rem;
        font-style: italic;
        opacity: 0.6;
      }
      .btn-verify {
        padding: 16px 32px;
        font-size: 1.1rem;
        font-weight: 700;
        border: none;
        border-radius: 12px;
        background: #6c5ce7;
        color: #fff;
        cursor: pointer;
        width: 100%;
        max-width: 480px;
      }
      .btn-verify:disabled {
        opacity: 0.5;
        cursor: not-allowed;
      }
    `,
  ],
})
export class SelfieComponent implements OnInit, OnDestroy {
  @ViewChild("videoEl", { static: true })
  videoRef!: ElementRef<HTMLVideoElement>;
  @ViewChild("canvasEl", { static: true })
  canvasRef!: ElementRef<HTMLCanvasElement>;

  cameraReady = false;
  loading = false;
  result: VerifyResponse | null = null;
  resultTier: "success" | "warning" | "error" = "error";

  private stream: MediaStream | null = null;

  constructor(private faceService: FaceVerifyService) {}

  ngOnInit(): void {
    this.startCamera();
  }

  ngOnDestroy(): void {
    // Libera a câmera ao sair do componente
    this.stream?.getTracks().forEach((t) => t.stop());
  }

  /** Inicia a câmera frontal. */
  private async startCamera(): Promise<void> {
    try {
      this.stream = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: "user",
          width: { ideal: 720 },
          height: { ideal: 1280 },
        },
        audio: false,
      });
      this.videoRef.nativeElement.srcObject = this.stream;
      this.cameraReady = true;
    } catch (err) {
      console.error("Câmera não disponível:", err);
    }
  }

  /** Captura o frame atual da <video> e retorna como Blob JPEG. */
  private captureFrame(): Promise<Blob> {
    const video = this.videoRef.nativeElement;
    const canvas = this.canvasRef.nativeElement;
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    const ctx = canvas.getContext("2d")!;
    // Espelha horizontalmente (mesma transformação do CSS)
    ctx.translate(canvas.width, 0);
    ctx.scale(-1, 1);
    ctx.drawImage(video, 0, 0);

    return new Promise<Blob>((resolve, reject) => {
      canvas.toBlob(
        (blob) =>
          blob ? resolve(blob) : reject(new Error("Falha ao gerar blob")),
        "image/jpeg",
        0.9,
      );
    });
  }

  /** Determina a tier (success | warning | error) com base no retorno. */
  private classifyResult(res: VerifyResponse): "success" | "warning" | "error" {
    if (res.match) return "success";
    if (res.confidence >= 0.5) return "warning";
    return "error";
  }

  /** Handler do botão "VALIDAR ACESSO". */
  async onVerify(): Promise<void> {
    this.loading = true;
    this.result = null;

    try {
      const blob = await this.captureFrame();

      // O service usa HttpClient (Observable), convertemos para Promise:
      const res = await this.faceService.verify(blob).toPromise();
      if (res) {
        this.result = res;
        this.resultTier = this.classifyResult(res);
      }
    } catch (err) {
      this.result = {
        match: false,
        name: "Erro",
        confidence: 0,
        message: "Falha de conexão com o servidor.",
      };
      this.resultTier = "error";
    } finally {
      this.loading = false;
    }
  }
}

// ─── 3. app.module.ts (adicionar imports necessários) ────────────
//
// import { HttpClientModule } from '@angular/common/http';
//
// @NgModule({
//   imports: [
//     HttpClientModule,
//     // ...outros imports
//   ],
//   declarations: [SelfieComponent],
// })
// export class AppModule {}
//
// ─── NOTAS PARA A EQUIPE ANGULAR ─────────────────────────────────
//
// 1. Ajuste `apiUrl` no service para apontar ao IP/domínio do servidor FastAPI.
// 2. Em produção, use HTTPS (necessário para câmera em dispositivos móveis).
// 3. O contrato de resposta do /verify é:
//    { match: boolean, name: string, confidence: number, message: string }
// 4. Três faixas de confiança:
//       > 0.65  → match=true  (Verde)
//    0.50–0.65  → match=false (Amarelo: "Tente novamente")
//       < 0.50  → match=false (Vermelho: "Desconhecido")
// 5. Endpoint GET /refresh-db  → força releitura do banco de fotos.
// 6. Endpoint GET /users       → lista nomes cadastrados (debug).
